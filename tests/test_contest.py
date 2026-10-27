import concurrent.futures
import json
import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path
from http.server import ThreadingHTTPServer
from contest.server import Store,Failure,Handler

class ContestTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'demo.db';self.store=Store(self.path)
        self.code=self.store.create_account();self.admincode=self.store.create_account('admin');self.code2=self.store.create_account()
        self.user=self.identify(self.code);self.admin=self.identify(self.admincode);self.user2=self.identify(self.code2)
        self.store.schedule(100,200,300,400)
    def tearDown(self):self.tmp.cleanup()
    def identify(self,code):
        token,_=self.store.login(code,100);return self.store.identity(token,101)
    def act(self,user,action,body,at,key=None):return self.store.act(user,action,body,key or __import__('secrets').token_hex(16),at)
    def idea(self,user=None,title='Clearer source dates'):
        return self.act(user or self.user,'submit',{'title':title,'problem':'The quote date is confusing.','solution':'Explain market date and generation date.','audience':'New visitors'},150)['id']
    def approve(self,id):return self.act(self.admin,'review',{'idea':id,'status':'approved'},160)
    def test_auth_rejection_expiry_and_unauthorized_moderation(self):
        with self.assertRaises(Failure):self.store.login('spoofed username',100)
        with self.assertRaises(Failure):self.act(None,'submit',{},150)
        with self.assertRaises(Failure):self.act(self.user,'review',{'idea':1,'status':'approved'},150)
        token,_=self.store.login(self.code,100)
        self.assertIsNone(self.store.identity(token,3700))
    def test_queue_private_duplicate_submission_and_review_reason(self):
        id=self.idea();self.assertFalse(self.store.state(None,160)['ideas']);self.assertEqual(len(self.store.state(self.admin,160)['queue']),1)
        self.assertEqual(self.idea(),id)
        with self.assertRaises(Failure):self.act(self.admin,'review',{'idea':id,'status':'rejected','reason':''},160)
        self.approve(id);self.assertEqual(len(self.store.state(None,170)['ideas']),1)
        self.assertFalse(self.store.state(self.user,170)['queue'])
    def test_vote_changes_and_late_retry_do_not_restore_old_choice(self):
        a=self.idea();b=self.idea(title='Accessible graph guide');self.approve(a);self.approve(b)
        key='same-original-request'
        self.act(self.user,'vote',{'idea':a},220,key);self.act(self.user,'vote',{'idea':b},221)
        self.act(self.user,'vote',{'idea':a},222,key)
        state=self.store.state(self.user,223);self.assertEqual(state['me']['vote'],b)
        self.assertEqual([x['total'] for x in state['ideas']],[0,1])
        with self.assertRaises(Failure):self.act(self.user,'vote',{'idea':b},224,key)
    def test_concurrent_requests_keep_one_active_vote(self):
        a=self.idea();b=self.idea(title='Clear units guide');self.approve(a);self.approve(b)
        def vote(i):return self.act(self.user,'vote',{'idea':a if i%2 else b},220)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(vote,range(24)))
        state=self.store.state(self.user,230);self.assertEqual(sum(i['total'] for i in state['ideas']),1)
    def test_boundaries_and_no_votes_result(self):
        body={'title':'A useful idea','problem':'A useful problem','solution':'A useful solution','audience':'Everyone'}
        for at in (99,200):
            with self.assertRaises(Failure):self.act(self.user,'submit',body,at)
        id=self.idea();self.approve(id)
        with self.assertRaises(Failure):self.act(self.user,'vote',{'idea':id},199)
        self.act(self.user,'vote',{'idea':id},200)
        with self.assertRaises(Failure):self.act(self.user2,'vote',{'idea':id},300)
        self.assertEqual(self.store.state(None,300)['round']['winner'],id)
    def test_runoff_tie_again_and_explicit_next_deadline(self):
        a=self.idea();b=self.idea(title='Better ratio labels');c=self.idea(self.user2,'Better chart contrast')
        for id in (a,b,c):self.approve(id)
        self.act(self.user,'vote',{'idea':a},210);self.act(self.user2,'vote',{'idea':b},210)
        self.assertEqual(self.store.state(None,300)['round']['phase'],'runoff')
        with self.assertRaises(Failure):self.act(self.user,'vote',{'idea':c},310)
        self.act(self.user,'vote',{'idea':a},310);self.act(self.user2,'vote',{'idea':b},310)
        self.assertEqual(self.store.state(None,400)['round']['phase'],'runoff_pending')
        with self.assertRaises(Failure):self.act(self.user,'vote',{'idea':a},410)
        self.act(self.admin,'runoff',{'deadline':600},410)
        self.act(self.user2,'vote',{'idea':a},500)
        state=self.store.state(None,600);self.assertEqual(state['round']['winner'],a)
        self.assertEqual(len(state['results']),7)
    def test_restart_preserves_ballots_sessions_and_frozen_results(self):
        id=self.idea();self.approve(id);self.act(self.user,'vote',{'idea':id},210)
        self.store=Store(self.path);self.assertEqual(self.store.state(self.user,220)['me']['vote'],id)
        state=self.store.state(None,300);self.store=Store(self.path)
        self.assertEqual(state['results'],self.store.state(None,301)['results'])
    def test_rate_limit_persists_and_resets_after_window(self):
        self.store.limit('test','submit',2,100);self.store.limit('test','submit',2,101)
        with self.assertRaises(Failure):Store(self.path).limit('test','submit',2,102)
        self.store.limit('test','submit',2,162)
    def test_plain_submitted_text_is_not_interpreted_by_store(self):
        payload='<img src=x onerror=alert(1)>'
        id=self.idea(title=payload);self.approve(id)
        self.assertEqual(self.store.state(None,170)['ideas'][0]['title'],payload)

class HttpTests(unittest.TestCase):
    def test_origin_csrf_cookie_private_paths_and_unauthorized_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            store=Store(Path(directory)/'test.db');code=store.create_account()
            class LocalHandler(Handler):pass
            LocalHandler.store=store
            server=ThreadingHTTPServer(('127.0.0.1',0),LocalHandler)
            threading.Thread(target=server.serve_forever,daemon=True).start()
            port=server.server_port;origin=f'http://127.0.0.1:{port}'
            def req(method,path,body=None,headers=None):
                conn=HTTPConnection('127.0.0.1',port);conn.request(method,path,json.dumps(body) if body is not None else None,headers or {});res=conn.getresponse();data=res.read();result=(res.status,dict(res.getheaders()),data);conn.close();return result
            try:
                hdr={'Origin':origin,'Content-Type':'application/json'}
                self.assertEqual(req('POST','/api/vote',{'idea':1},hdr)[0],401)
                self.assertEqual(req('POST','/api/login',{'code':code},{**hdr,'Origin':'https://evil.example'})[0],403)
                status,headers,_=req('POST','/api/login',{'code':code},hdr);self.assertEqual(status,200)
                self.assertIn('HttpOnly',headers['Set-Cookie']);cookie=headers['Set-Cookie'].split(';')[0]
                self.assertEqual(req('POST','/api/vote',{'idea':1},{**hdr,'Cookie':cookie})[0],403)
                self.assertEqual(req('GET','/contest/server.py')[0],404)
                self.assertEqual(req('GET','/.env')[0],404)
                self.assertEqual(req('GET','/api/contest')[0],200)
            finally:server.shutdown();server.server_close()

if __name__=='__main__':unittest.main()
