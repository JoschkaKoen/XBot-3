"""Regression checks use fake publishing and providers; no X/AI calls."""
import json
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from services.age_metrics import due_records, snapshot, report
from services.content_quality import issues, reviewed_choice, verified_cefr, used_words
from services.delivery import publish_once, DeliveryPending
from utils.retry import with_retry
from utils.runtime import Health, ConsoleLog
from utils.schedule import startup_due_at

class ReliabilityTests(unittest.TestCase):
    def test_402_not_retried(self):
        class Billing(Exception):
            status_code = 402
        calls = []
        @with_retry(max_attempts=6, base_delay=0)
        def write():
            calls.append(1)
            raise Billing()
        with self.assertRaises(Billing):
            write()
        self.assertEqual(len(calls), 1)

    def test_successful_write_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            def send(*args):
                calls.append(1)
                return '123', 'https://x.com/i/web/status/123'
            for _ in range(2):
                self.assertEqual(publish_once('primary', 7, 'text', 'video', send, root)[0], '123')
            self.assertEqual(len(calls), 1)
            health = Health(Path(root)/'health.json').read()
            self.assertEqual(health['channels']['x:primary']['status'], 'ok')

    def test_confirmed_write_recovers_history_after_crash(self):
        from services.delivery import recover_primary_history
        with tempfile.TemporaryDirectory() as root:
            publish_once('primary', 7, 'text', 'video', lambda *_: ('123', 'url'), root,
                         record={'cycle': 7, 'source_word': 'Begegnung'})
            recover_primary_history(root)
            recover_primary_history(root)
            rows = json.loads((Path(root)/'post_history.json').read_text())
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['source_word'], 'Begegnung')

    def test_uncertain_write_is_held_across_cycles(self):
        with tempfile.TemporaryDirectory() as root:
            def send(*args):
                raise ConnectionError('response lost')
            with self.assertRaises(ConnectionError):
                publish_once('zh', 7, 'text', 'video', send, root)
            for cycle in (7, 8):
                with self.assertRaises(DeliveryPending):
                    publish_once('zh', cycle, 'text', 'video', lambda *_: self.fail('duplicate'), root)

    def test_billing_isolated_from_website_status(self):
        class Billing(Exception):
            status_code = 402
        with tempfile.TemporaryDirectory() as root:
            def send(*args):
                raise Billing()
            with self.assertRaises(Billing):
                publish_once('zh', 7, 'text', 'video', send, root)
            health = Health(Path(root)/'health.json')
            health.delivery('website:zh', 'ok')
            state = health.read()['channels']
            self.assertEqual(state['x:zh']['failure'], 'billing')
            self.assertEqual(state['website:zh']['status'], 'ok')

    def test_crash_before_write_and_corrupt_ledger_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            f = Path(root)/'delivery-primary.json'
            f.write_text('{corrupt')
            with self.assertRaises(ValueError):
                publish_once('primary', 1, 'text', 'video', lambda *_: self.fail('write'), root)

    def test_single_instance_across_processes(self):
        with tempfile.TemporaryDirectory() as root:
            lock = str(Path(root)/'bot.lock')
            owner = subprocess.Popen([sys.executable, '-u', '-c',
                'from utils.runtime import acquire_instance;import sys;acquire_instance(sys.argv[1]);print("ready",flush=True);sys.stdin.read()', lock],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(owner.stdout.readline().strip(), 'ready')
                child = subprocess.run([sys.executable, '-c',
                    'from utils.runtime import acquire_instance;import sys;acquire_instance(sys.argv[1])', lock],
                    capture_output=True, text=True)
                self.assertNotEqual(child.returncode, 0)
            finally:
                owner.communicate('', timeout=5)

    def test_startup_preserves_interval_and_scheduled_slot(self):
        now = datetime(2026, 10, 7, 5, 30, tzinfo=timezone.utc)
        history = [{'tweet_id':'1','timestamp':(now-timedelta(hours=1)).isoformat()}]
        self.assertEqual(startup_due_at({}, history, False, '', 21600, now), now.timestamp()+18000)
        self.assertEqual(startup_due_at({}, history, True, '04:00,08:00',21600,now),
                         now.replace(hour=8,minute=0).timestamp())
        future = now.timestamp()+200
        self.assertEqual(startup_due_at({'next_post_at':future},history,True,'08:00',21600,now),future)

    def test_fixed_age_windows_budget_and_missing_data(self):
        now = datetime(2026,10,7,tzinfo=timezone.utc)
        rows = [{'tweet_id':str(i),'timestamp':(now-timedelta(hours=age)).isoformat()}
                for i,age in enumerate((2,25,170,400))]
        selected = due_records(rows,1,now)
        self.assertEqual(len(selected),1)
        self.assertEqual(selected[0]['tweet_id'],'1')
        updated = snapshot(selected[0],{'impression_count':15},now)
        self.assertEqual(updated['age_metrics']['24h']['age_hours'],25)
        self.assertFalse(due_records([updated],3,now))
        self.assertIsNone(report(rows)['7d']['median_impressions'])

    def test_quality_examples_and_valid_inflections(self):
        self.assertTrue(issues('kochen','Zum kochen brauche ich Mut.'))
        self.assertTrue(issues('ausprobieren','Ich will Kuchen ausprobieren bevor meine Diät beginnt.'))
        self.assertTrue(issues('hold','I am on hold.',language='English'))
        self.assertTrue(issues('treat','One cake treat.', '蛋糕享受', 'English'))
        self.assertFalse(issues('bereichern','Urlaub bereichert die Seele.'))
        self.assertFalse(issues('procrastinate','She keeps procrastinating.',language='English'))
        self.assertTrue(issues('hold','This household is quiet.',language='English'))
        self.assertEqual(used_words([{'source_word':' DIE Gemeinschaft '}],'German'),{'gemeinschaft'})

    def test_quality_review_never_falls_back_to_bad_candidate(self):
        for raw in ('1 — funny', '{"approved":false,"best_index":0}', '{"approved":true,"best_index":99}'):
            with self.assertRaises(ValueError):
                reviewed_choice(raw,[{}])
        self.assertEqual(reviewed_choice('{"approved":true,"best_index":0}', [{'ok':1}]),{'ok':1})

    def test_cefr_requires_reviewed_source(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'lexicon.json'
            path.write_text(json.dumps({'english':{'picnic':{'cefr':'C1'}}}))
            with patch.dict(os.environ,{'BOT_CEFR_LEXICON':str(path)}):
                self.assertEqual(verified_cefr('picnic','English'),'')

    def test_console_rotates_without_ansi_or_spinners(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'console.log'
            stream=ConsoleLog(path,max_bytes=50,backups=2)
            stream.write('\rspinner')
            for _ in range(10):
                stream.write('\x1b[91mhello world\x1b[0m\n')
            stream.flush()
            stream.handler.close()
            self.assertLessEqual(len(list(Path(root).iterdir())),3)
            for file in Path(root).iterdir():
                self.assertNotIn('spinner',file.read_text())
                self.assertNotIn('\x1b',file.read_text())

if __name__ == '__main__':
    unittest.main()
