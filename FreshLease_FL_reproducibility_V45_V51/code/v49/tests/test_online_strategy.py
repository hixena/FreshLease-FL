"""Exercise the strategy boundary without Docker or Flower installation."""

from __future__ import annotations

import csv
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from flower_prototype.access_control import AccessLedger, public_key_b64, sign_payload
from flower_prototype.real_data import (
    client_partition, initial_parameters, local_train, parameter_digest,
    sign_flip_update,
)


class FakeFedAvg:
    def __init__(self, **kwargs):
        pass

    def configure_fit(self, server_round, parameters, client_manager):
        return [(proxy, object()) for proxy in client_manager]

    configure_evaluate = configure_fit

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        weights = np.asarray([res.num_examples for _, res in results])
        updates = [res.parameters for _, res in results]
        return [
            sum(float(w) * u[i] for w, u in zip(weights, updates)) / weights.sum()
            for i in range(len(updates[0]))
        ], {}


class OnlineStrategyTests(unittest.TestCase):
    def test_moderate_norm_is_soft_rejected_then_repeated_excess_revokes(self):
        fake = types.SimpleNamespace(
            server=types.SimpleNamespace(strategy=types.SimpleNamespace(FedAvg=FakeFedAvg)),
            common=types.SimpleNamespace(
                parameters_to_ndarrays=lambda values: values,
                ndarrays_to_parameters=lambda values: values,
            ),
        )
        request_stub = types.SimpleNamespace(RequestException=Exception)
        with patch.dict(sys.modules, {"flwr": fake, "requests": request_stub}):
            module = importlib.import_module("flower_prototype.server")
        x_validation, y_validation, _, _ = module.load_validation_test_sets()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = AccessLedger(
                root / "ledger.sqlite", root / "sign.pem",
                mechanism_variant="oracle_benign",
            )
            private_key = Ed25519PrivateKey.generate()
            public_key = public_key_b64(private_key)
            registration = {
                "action": "REGISTER", "node_id": "node",
                "profile": "honest", "public_key": public_key,
            }
            ledger.register_node(
                "node", "honest", public_key,
                sign_payload(private_key, registration),
            )
            base = initial_parameters()
            proxy = types.SimpleNamespace(cid="cid")

            class Manager(list):
                def num_available(self):
                    return len(self)

            strategy = module.RevalidatingFedAvg(
                x_validation, y_validation, "http://unused", "token", enabled=True,
                expected_initial_clients=1,
                round_node_events_path=root / "round_nodes.csv",
            )

            def controller(path, values):
                if path.endswith("/verify"):
                    return ledger.verify_fit_commitment(**values)
                if path.endswith("/revoke"):
                    return ledger.revoke_training_access(**values)
                if path.endswith("/accept"):
                    return ledger.accept_fit_update(**values)
                if path.endswith("/aggregated"):
                    return ledger.record_fit_aggregation(**values)
                raise AssertionError(path)

            strategy._post = controller
            for server_round in (1, 2):
                update = [value.copy() for value in base]
                commitment = {
                    "action": "FIT_COMMITMENT", "node_id": "node",
                    "server_round": server_round,
                    "update_hash": parameter_digest(update),
                    "parent_hash": parameter_digest(base),
                }
                result = types.SimpleNamespace(
                    parameters=update, num_examples=1,
                    metrics={
                        **commitment,
                        "fit_signature": sign_payload(private_key, commitment),
                    },
                )
                strategy.configure_fit(server_round, base, Manager([proxy]))
                with patch.object(
                    module, "classify_update_norm",
                    return_value=("MODERATE_EXCESS", 0.5, 1.0, 1.5),
                ):
                    aggregated, _ = strategy.aggregate_fit(
                        server_round, [(proxy, result)], [],
                    )
                self.assertIsNone(aggregated)
                if server_round == 1:
                    self.assertIsNone(ledger.node_status("node")["revoked_round"])
                    self.assertNotIn(proxy.cid, strategy.rejected_cids)
                else:
                    self.assertEqual(ledger.node_status("node")["revoked_round"], 2)

            with (root / "round_nodes.csv").open(newline="", encoding="utf-8") as handle:
                events = list(csv.DictReader(handle))
            self.assertEqual(events[0]["reason"], "EXCESS_UPDATE_NORM_REJECTED")
            self.assertEqual(events[0]["norm_screening_outcome"], "MODERATE_REJECTED")
            self.assertEqual(events[0]["norm_strike_count"], "1")
            self.assertEqual(events[0]["aggregated"], "0")
            self.assertEqual(events[1]["reason"], "EXCESS_UPDATE_NORM")
            self.assertEqual(events[1]["norm_screening_outcome"], "REPEATED_REVOKED")
            self.assertEqual(events[1]["norm_strike_count"], "2")
            ledger.connection.close()

    def test_finite_validation_loss_is_telemetry_not_client_exclusion(self):
        fake = types.SimpleNamespace(
            server=types.SimpleNamespace(strategy=types.SimpleNamespace(FedAvg=FakeFedAvg)),
            common=types.SimpleNamespace(
                parameters_to_ndarrays=lambda values: values,
                ndarrays_to_parameters=lambda values: values,
            ),
        )
        request_stub = types.SimpleNamespace(RequestException=Exception)
        with patch.dict(sys.modules, {"flwr": fake, "requests": request_stub}):
            module = importlib.import_module("flower_prototype.server")
        x_validation, y_validation, _, _ = module.load_validation_test_sets()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = AccessLedger(
                root / "ledger.sqlite", root / "sign.pem",
                mechanism_variant="oracle_benign",
            )
            private_key = Ed25519PrivateKey.generate()
            public_key = public_key_b64(private_key)
            registration = {
                "action": "REGISTER", "node_id": "honest-node",
                "profile": "honest", "public_key": public_key,
            }
            ledger.register_node(
                "honest-node", "honest", public_key,
                sign_payload(private_key, registration),
            )
            base = initial_parameters()
            update = [value.copy() for value in base]
            commitment = {
                "action": "FIT_COMMITMENT", "node_id": "honest-node",
                "server_round": 1, "update_hash": parameter_digest(update),
                "parent_hash": parameter_digest(base),
            }
            result = types.SimpleNamespace(
                parameters=update, num_examples=1,
                metrics={
                    **commitment,
                    "fit_signature": sign_payload(private_key, commitment),
                },
            )
            proxy = types.SimpleNamespace(cid="honest-cid")

            class Manager(list):
                def num_available(self):
                    return len(self)

            report = root / "round_nodes.csv"
            strategy = module.RevalidatingFedAvg(
                x_validation, y_validation, "http://unused", "token", enabled=True,
                expected_initial_clients=1, round_node_events_path=report,
            )
            strategy.norm_strikes["honest-node"] = 1

            def controller(path, values):
                if path.endswith("/verify"):
                    return ledger.verify_fit_commitment(**values)
                if path.endswith("/revoke"):
                    return ledger.revoke_training_access(**values)
                if path.endswith("/accept"):
                    return ledger.accept_fit_update(**values)
                if path.endswith("/aggregated"):
                    return ledger.record_fit_aggregation(**values)
                raise AssertionError(path)

            strategy._post = controller
            strategy.configure_fit(1, base, Manager([proxy]))
            with patch.object(module, "screen_update", return_value="VALIDATION_LOSS"):
                aggregated, _ = strategy.aggregate_fit(1, [(proxy, result)], [])
            self.assertIsNotNone(aggregated)
            status = ledger.node_status("honest-node")
            self.assertEqual(status["access_state"], "ADMITTED")
            self.assertIsNone(status["revoked_round"])
            self.assertNotIn(proxy.cid, strategy.rejected_cids)
            self.assertEqual(strategy.norm_strikes["honest-node"], 0)
            with report.open(newline="", encoding="utf-8") as handle:
                event = next(csv.DictReader(handle))
            self.assertEqual(event["reason"], "")
            self.assertEqual(event["training_evidence_outcome"], "VERIFIED_CLEAN")
            self.assertEqual(event["aggregated"], "1")
            self.assertEqual(event["validation_loss_flag"], "1")
            self.assertNotEqual(event["candidate_validation_loss"], "")
            self.assertNotEqual(event["global_validation_loss"], "")
            ledger.connection.close()

    def test_attack_is_revoked_before_aggregation_and_excluded_next_round(self):
        fake = types.SimpleNamespace(
            server=types.SimpleNamespace(strategy=types.SimpleNamespace(FedAvg=FakeFedAvg)),
            common=types.SimpleNamespace(
                parameters_to_ndarrays=lambda values: values,
                ndarrays_to_parameters=lambda values: values,
            ),
        )
        request_stub = types.SimpleNamespace(
            RequestException=Exception,
            post=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected HTTP")),
        )
        with patch.dict(sys.modules, {"flwr": fake, "requests": request_stub}):
            strategy_module = importlib.import_module("flower_prototype.server")
        test_x, test_y, _, _ = strategy_module.load_validation_test_sets()
        for seed in range(3):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as directory:
                directory_path = Path(directory)
                ledger = AccessLedger(
                    directory_path / "ledger.sqlite", directory_path / "sign.pem",
                    mechanism_variant="oracle_benign",
                )
                base = initial_parameters()
                strategy = strategy_module.RevalidatingFedAvg(
                    test_x, test_y, "http://unused", "token", enabled=True,
                    expected_initial_clients=4,
                    round_node_events_path=directory_path / "round_nodes.csv",
                )
                def controller(path, values):
                    if path.endswith("/verify"):
                        return ledger.verify_fit_commitment(**values)
                    if path.endswith("/revoke"):
                        return ledger.revoke_training_access(**values)
                    if path.endswith("/accept"):
                        return ledger.accept_fit_update(**values)
                    if path.endswith("/aggregated"):
                        return ledger.record_fit_aggregation(**values)
                    raise AssertionError(path)
                strategy._post = controller
                proxies = [types.SimpleNamespace(cid=str(pid)) for pid in (0, 1, 2, 4)]
                class Manager(list):
                    def num_available(self):
                        return len(self)

                manager = Manager(proxies)
                strategy.configure_fit(1, base, manager)
                fit_results = []
                for pid, proxy in zip((0, 1, 2, 4), proxies):
                    node_id = f"node-{pid}"
                    private_key = Ed25519PrivateKey.generate()
                    public_key = public_key_b64(private_key)
                    profile = "honest" if pid != 4 else "diverse_then_repeat_farming"
                    registration = {
                        "action": "REGISTER", "node_id": node_id,
                        "profile": profile, "public_key": public_key,
                    }
                    ledger.register_node(
                        node_id, profile, public_key,
                        sign_payload(private_key, registration),
                    )
                    x, y = client_partition(pid, 6, 0.5, seed)
                    updated = local_train(
                        base, x, y, epochs=2, learning_rate=0.15,
                        batch_size=32, seed=seed * 10000 + pid * 100 + 1,
                    )
                    if pid == 4:
                        # Keep this test focused on an unambiguous hard norm
                        # failure; finite loss-only regressions are covered by
                        # the soft-rejection test above.
                        updated = sign_flip_update(base, updated, 10.0)
                    commitment = {
                        "action": "FIT_COMMITMENT", "node_id": node_id,
                        "server_round": 1, "update_hash": parameter_digest(updated),
                        "parent_hash": parameter_digest(base),
                    }
                    result = types.SimpleNamespace(
                        parameters=updated, num_examples=len(y),
                        metrics={**commitment, "fit_signature": sign_payload(private_key, commitment)},
                    )
                    fit_results.append((proxy, result))
                aggregated, _ = strategy.aggregate_fit(1, fit_results, [])
                self.assertIsNotNone(aggregated)
                self.assertEqual(ledger.node_status("node-4")["revoked_round"], 1)
                self.assertEqual(
                    ledger.experiment_summary()["nodes"][-1]["aggregated_fit_events"], 0
                )
                self.assertEqual(len(strategy.configure_fit(2, aggregated, manager)), 3)
                incomplete = Manager([proxies[0], proxies[1], proxies[3]])
                with self.assertRaisesRegex(RuntimeError, "round 3 sampled 2 clients; expected 3"):
                    strategy.configure_fit(3, aggregated, incomplete)
                with (directory_path / "round_nodes.csv").open(newline="", encoding="utf-8") as handle:
                    records = list(csv.DictReader(handle))
                self.assertEqual(len(records), 4)
                self.assertTrue(all(row["selected"] == "1" and row["returned"] == "1"
                                    for row in records))
                self.assertEqual(sum(int(row["aggregated"]) for row in records), 3)
                self.assertEqual(next(row["reason"] for row in records
                                      if row["node_id"] == "node-4"),
                                 ledger.node_status("node-4")["revocation_reason"])
                self.assertTrue(ledger.verify_integrity()["valid"])
                ledger.connection.close()

    def test_initial_join_barrier_waits_for_all_and_rejects_partial_sample(self):
        fake = types.SimpleNamespace(
            server=types.SimpleNamespace(strategy=types.SimpleNamespace(FedAvg=FakeFedAvg)),
            common=types.SimpleNamespace(
                parameters_to_ndarrays=lambda values: values,
                ndarrays_to_parameters=lambda values: values,
            ),
        )
        request_stub = types.SimpleNamespace(RequestException=Exception)
        with patch.dict(sys.modules, {"flwr": fake, "requests": request_stub}):
            module = importlib.import_module("flower_prototype.server")
        validation_x, validation_y, _, _ = module.load_validation_test_sets()
        proxies = [types.SimpleNamespace(cid=str(i)) for i in range(4)]

        class JoiningManager(list):
            def __init__(self):
                super().__init__(proxies)
                self.polls = 0

            def num_available(self):
                self.polls += 1
                return 3 if self.polls < 3 else 4

        manager = JoiningManager()
        strategy = module.RevalidatingFedAvg(
            validation_x, validation_y, "http://unused", "token", enabled=True,
            expected_initial_clients=4, initial_join_timeout=1.0,
        )
        with patch.object(module.time, "sleep", return_value=None):
            selected = strategy.configure_fit(1, initial_parameters(), manager)
        self.assertGreaterEqual(manager.polls, 3)
        self.assertEqual(len(selected), 4)

        unavailable = types.SimpleNamespace(num_available=lambda: 3)
        with self.assertRaisesRegex(RuntimeError, "available=3, required=4"):
            module.wait_for_initial_clients(unavailable, 4, 0.001)

        another = module.RevalidatingFedAvg(
            validation_x, validation_y, "http://unused", "token", enabled=True,
            expected_initial_clients=4,
        )
        with patch.object(FakeFedAvg, "configure_fit", return_value=[
            (proxy, object()) for proxy in proxies[:3]
        ]):
            with self.assertRaisesRegex(RuntimeError, "round 1 sampled 3 clients"):
                another.configure_fit(1, initial_parameters(), manager)

        exhausted = module.RevalidatingFedAvg(
            validation_x, validation_y, "http://unused", "token", enabled=True,
            expected_initial_clients=1,
        )
        exhausted.rejected_cids.add("only-client")
        self.assertEqual(exhausted.configure_fit(2, initial_parameters(), manager), [])

    def test_selected_client_without_update_remains_visible_in_round_ledger(self):
        fake = types.SimpleNamespace(
            server=types.SimpleNamespace(strategy=types.SimpleNamespace(FedAvg=FakeFedAvg)),
            common=types.SimpleNamespace(
                parameters_to_ndarrays=lambda values: values,
                ndarrays_to_parameters=lambda values: values,
            ),
        )
        request_stub = types.SimpleNamespace(RequestException=Exception)
        with patch.dict(sys.modules, {"flwr": fake, "requests": request_stub}):
            module = importlib.import_module("flower_prototype.server")
        x_validation, y_validation, _, _ = module.load_validation_test_sets()
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "selected.csv"
            strategy = module.RevalidatingFedAvg(
                x_validation, y_validation, "http://unused", "token", enabled=True,
                round_node_events_path=report,
            )
            selected = [types.SimpleNamespace(cid="absent")]
            strategy.configure_fit(1, initial_parameters(), selected)
            self.assertEqual(strategy.aggregate_fit(1, [], ["unreachable"])[0], None)
            with report.open(newline="", encoding="utf-8") as handle:
                event = next(csv.DictReader(handle))
            self.assertEqual(event["selected"], "1")
            self.assertEqual(event["returned"], "0")
            self.assertEqual(event["node_id"], "")
            self.assertEqual(event["reason"], "NO_RESPONSE")


if __name__ == "__main__":
    unittest.main()
