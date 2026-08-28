import unittest
import warnings

from pobot.config import RiskConfig
from pobot.risk import RiskManager, RiskState


class TestRiskManager(unittest.TestCase):
    def test_fixed_stake_is_constant(self):
        manager = RiskManager(RiskConfig(stake_mode="fixed", fixed_stake=5.0), payout=0.85)
        state = RiskState(balance=1000, starting_balance=1000)
        self.assertEqual(manager.next_stake(state), 5.0)

    def test_fraction_stake_scales_with_balance(self):
        manager = RiskManager(RiskConfig(stake_mode="fraction", fraction=0.02), payout=0.85)
        state = RiskState(balance=1000, starting_balance=1000)
        self.assertAlmostEqual(manager.next_stake(state), 20.0)

    def test_kelly_stake_requires_estimated_winrate(self):
        manager = RiskManager(RiskConfig(stake_mode="kelly"), payout=0.85)
        state = RiskState(balance=1000, starting_balance=1000)
        with self.assertRaises(ValueError):
            manager.next_stake(state)

    def test_kelly_stake_uses_capped_fraction(self):
        manager = RiskManager(
            RiskConfig(stake_mode="kelly", kelly_cap=0.1), payout=0.85, estimated_winrate=0.65
        )
        state = RiskState(balance=1000, starting_balance=1000)
        stake = manager.next_stake(state)
        self.assertLessEqual(stake, 100.0)  # 0.1 * 1000
        self.assertGreater(stake, 0.0)

    def test_martingale_multiplies_after_losses(self):
        config = RiskConfig(
            stake_mode="fixed", fixed_stake=1.0, martingale_enabled=True,
            martingale_multiplier=2.0, martingale_max_steps=3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            manager = RiskManager(config, payout=0.85)
        state = RiskState(balance=1000, starting_balance=1000, consecutive_losses=2)
        self.assertAlmostEqual(manager.next_stake(state), 4.0)  # 1.0 * 2^2

    def test_martingale_caps_at_max_steps(self):
        config = RiskConfig(
            stake_mode="fixed", fixed_stake=1.0, martingale_enabled=True,
            martingale_multiplier=2.0, martingale_max_steps=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            manager = RiskManager(config, payout=0.85)
        state = RiskState(balance=1000, starting_balance=1000, consecutive_losses=10)
        self.assertAlmostEqual(manager.next_stake(state), 4.0)  # 1.0 * 2^2, no 2^10

    def test_martingale_enabled_emits_warning(self):
        config = RiskConfig(stake_mode="fixed", martingale_enabled=True)
        with self.assertWarns(UserWarning):
            RiskManager(config, payout=0.85)

    def test_max_trades_per_day_blocks_trading(self):
        manager = RiskManager(RiskConfig(max_trades_per_day=3), payout=0.85)
        state = RiskState(balance=1000, starting_balance=1000, trades_today=3)
        can, reason = manager.can_trade(state)
        self.assertFalse(can)
        self.assertIn("diarias", reason)

    def test_daily_loss_limit_blocks_trading(self):
        manager = RiskManager(RiskConfig(daily_loss_limit=50.0), payout=0.85)
        state = RiskState(balance=950, starting_balance=1000, pnl_today=-50.0)
        can, _ = manager.can_trade(state)
        self.assertFalse(can)

    def test_daily_profit_target_blocks_trading(self):
        manager = RiskManager(RiskConfig(daily_profit_target=100.0), payout=0.85)
        state = RiskState(balance=1100, starting_balance=1000, pnl_today=100.0)
        can, _ = manager.can_trade(state)
        self.assertFalse(can)

    def test_can_trade_true_within_limits(self):
        manager = RiskManager(RiskConfig(max_trades_per_day=10, daily_loss_limit=50.0), payout=0.85)
        state = RiskState(balance=1000, starting_balance=1000, trades_today=2, pnl_today=-10.0)
        can, _ = manager.can_trade(state)
        self.assertTrue(can)

    def test_register_trade_updates_state(self):
        state = RiskState(balance=1000, starting_balance=1000)
        state.register_trade(pnl=10.0, was_loss=False)
        self.assertEqual(state.balance, 1010.0)
        self.assertEqual(state.trades_today, 1)
        self.assertEqual(state.consecutive_losses, 0)
        state.register_trade(pnl=-5.0, was_loss=True)
        self.assertEqual(state.consecutive_losses, 1)

    def test_unknown_stake_mode_raises(self):
        manager = RiskManager(RiskConfig(stake_mode="bogus"), payout=0.85)
        state = RiskState(balance=1000, starting_balance=1000)
        with self.assertRaises(ValueError):
            manager.next_stake(state)


if __name__ == "__main__":
    unittest.main()
