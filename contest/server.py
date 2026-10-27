"""Loopback-only contest demonstration. This is not a production identity service."""
import argparse
import hashlib
import hmac
import json
import secrets
import sqlite3
import time
from contextlib import contextmanager
from http.cookies import SimpleCookie
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class Failure(Exception):
    def __init__(self, status, code): self.status, self.code = status, code

class Store:
    def __init__(self, path):
        self.path = str(path)
        with self.db() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS accounts(id INTEGER PRIMARY KEY, code TEXT UNIQUE, role TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY, account INTEGER REFERENCES accounts, csrf TEXT, expires INTEGER);
            CREATE TABLE IF NOT EXISTS round(id INTEGER PRIMARY KEY CHECK(id=1), phase TEXT, opens INTEGER, submits INTEGER, closes INTEGER, runoff_closes INTEGER, ballot TEXT DEFAULT 'main', winner INTEGER, outcome TEXT DEFAULT '');
            INSERT OR IGNORE INTO round(id,phase) VALUES(1,'draft');
            CREATE TABLE IF NOT EXISTS ideas(id INTEGER PRIMARY KEY, author INTEGER REFERENCES accounts, title TEXT, problem TEXT, solution TEXT, audience TEXT, status TEXT DEFAULT 'pending', reason TEXT DEFAULT '');
            CREATE TABLE IF NOT EXISTS votes(account INTEGER REFERENCES accounts, ballot TEXT, idea INTEGER REFERENCES ideas, PRIMARY KEY(account,ballot));
            CREATE TABLE IF NOT EXISTS candidates(ballot TEXT, idea INTEGER, PRIMARY KEY(ballot,idea));
            CREATE TABLE IF NOT EXISTS results(ballot TEXT, idea INTEGER, total INTEGER, PRIMARY KEY(ballot,idea));
            CREATE TABLE IF NOT EXISTS requests(account INTEGER, key TEXT, digest TEXT, response TEXT, PRIMARY KEY(account,key));
            CREATE TABLE IF NOT EXISTS limits(subject TEXT, action TEXT, at INTEGER);
            CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, account INTEGER, action TEXT, idea INTEGER, reason TEXT, at INTEGER);
            ''')
    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally: db.close()
    def create_account(self, role='member'):
        code = secrets.token_urlsafe(24)
        with self.db() as db:
            db.execute('INSERT INTO accounts(code,role) VALUES(?,?)',(hashlib.sha256(code.encode()).hexdigest(),role))
        return code
    def limit(self, subject, action, maximum, now):
        with self.db() as db:
            db.execute('DELETE FROM limits WHERE at<?',(now-3600,))
            count=db.execute('SELECT count(*) FROM limits WHERE subject=? AND action=? AND at>?',(str(subject),action,now-60)).fetchone()[0]
            if count>=maximum: raise Failure(429,'rate')
            db.execute('INSERT INTO limits VALUES(?,?,?)',(str(subject),action,now))
    def login(self, code, now):
        with self.db() as db:
            account=db.execute('SELECT * FROM accounts WHERE code=?',(hashlib.sha256(str(code).encode()).hexdigest(),)).fetchone()
            if not account: raise Failure(401,'auth')
            token,csrf=secrets.token_urlsafe(32),secrets.token_urlsafe(24)
            db.execute('INSERT INTO sessions VALUES(?,?,?,?)',(hashlib.sha256(token.encode()).hexdigest(),account['id'],csrf,now+3600))
            return token,csrf
    def identity(self, token, now):
        with self.db() as db:
            row=db.execute('SELECT a.id,a.role,s.csrf FROM sessions s JOIN accounts a ON a.id=s.account WHERE s.token=? AND expires>?',(hashlib.sha256(token.encode()).hexdigest(),now)).fetchone()
            return dict(row) if row else None
    def schedule(self, opens, submits, closes, runoff_closes):
        if not all(type(x) is int for x in (opens,submits,closes,runoff_closes)) or not opens<submits<closes<runoff_closes: raise Failure(400,'dates')
        with self.db() as db:
            if db.execute('SELECT phase FROM round').fetchone()[0]!='draft': raise Failure(409,'phase')
            db.execute("UPDATE round SET opens=?,submits=?,closes=?,runoff_closes=?,phase='scheduled'",(opens,submits,closes,runoff_closes))
    def _counts(self, db, ballot):
        if ballot=='main': ids=db.execute("SELECT id FROM ideas WHERE status='approved' ORDER BY id").fetchall()
        else: ids=db.execute('SELECT idea FROM candidates WHERE ballot=? ORDER BY idea',(ballot,)).fetchall()
        return [{'id':r[0],'total':db.execute('SELECT count(*) FROM votes WHERE ballot=? AND idea=?',(ballot,r[0])).fetchone()[0]} for r in ids]
    def _settle(self, db, r):
        counts=self._counts(db,r['ballot'])
        for c in counts: db.execute('INSERT OR IGNORE INTO results VALUES(?,?,?)',(r['ballot'],c['id'],c['total']))
        highest=max((c['total'] for c in counts),default=0)
        tied=[c['id'] for c in counts if c['total']==highest]
        if highest==0:
            db.execute("UPDATE round SET phase='completed',outcome='no_votes'")
        elif len(tied)==1:
            db.execute("UPDATE round SET phase='completed',winner=?,outcome='winner'",(tied[0],))
        else:
            next_ballot='runoff-1' if r['ballot']=='main' else 'runoff-'+str(int(r['ballot'].split('-')[1])+1)
            for idea in tied: db.execute('INSERT INTO candidates VALUES(?,?)',(next_ballot,idea))
            # Repeated ties wait for a newly announced deadline; never draw lots.
            phase='runoff' if r['ballot']=='main' else 'runoff_pending'
            db.execute('UPDATE round SET phase=?,ballot=?',(phase,next_ballot))
    def _advance(self, db, now):
        r=dict(db.execute('SELECT * FROM round').fetchone())
        if r['phase']=='scheduled':
            phase='draft' if now<r['opens'] else 'submissions' if now<r['submits'] else 'voting'
            if phase!='draft': db.execute('UPDATE round SET phase=?',(phase,))
        elif r['phase']=='submissions' and now>=r['submits']: db.execute("UPDATE round SET phase='voting'")
        r=dict(db.execute('SELECT * FROM round').fetchone())
        if r['phase']=='voting' and now>=r['closes']: self._settle(db,r)
        r=dict(db.execute('SELECT * FROM round').fetchone())
        if r['phase']=='runoff' and now>=r['runoff_closes']: self._settle(db,r)
        r=dict(db.execute('SELECT * FROM round').fetchone())
        if r['phase']=='scheduled': r['phase']='draft'
        return r
    def state(self, identity, now):
        with self.db() as db:
            r=self._advance(db,now)
            ideas=[dict(x) for x in db.execute("SELECT id,title,problem,solution,audience FROM ideas WHERE status='approved' ORDER BY id")]
            counts={c['id']:c['total'] for c in self._counts(db,r['ballot'])}
            allowed={x[0] for x in db.execute('SELECT idea FROM candidates WHERE ballot=?',(r['ballot'],))}
            for idea in ideas: idea.update(total=counts.get(idea['id'],0),eligible=r['ballot']=='main' or idea['id'] in allowed)
            me=None; queue=[]; own=[]
            if identity:
                me={'role':identity['role'],'vote':None}
                row=db.execute('SELECT idea FROM votes WHERE account=? AND ballot=?',(identity['id'],r['ballot'])).fetchone()
                me['vote']=row[0] if row else None
                own=[dict(x) for x in db.execute('SELECT id,title,status,reason FROM ideas WHERE author=?',(identity['id'],))]
                if identity['role']=='admin': queue=[dict(x) for x in db.execute("SELECT id,title,problem,solution,audience FROM ideas WHERE status='pending' ORDER BY id")]
            return {'demo':True,'round':r,'ideas':ideas,'me':me,'queue':queue,'own':own,'results':[dict(x) for x in db.execute('SELECT * FROM results ORDER BY ballot,idea')],'server_time':now,'timezone':'Asia/Jerusalem'}
    def act(self, user, action, body, key, now):
        if not user: raise Failure(401,'auth')
        if not isinstance(key,str) or not 16<=len(key)<=100: raise Failure(400,'request')
        digest=hashlib.sha256(json.dumps([action,body],sort_keys=True).encode()).hexdigest()
        with self.db() as db:
            r=self._advance(db,now)
            previous=db.execute('SELECT * FROM requests WHERE account=? AND key=?',(user['id'],key)).fetchone()
            if previous:
                if previous['digest']!=digest: raise Failure(409,'request')
                return json.loads(previous['response'])
            result={'ok':True}
            if action=='submit':
                if r['phase']!='submissions': raise Failure(409,'phase')
                fields=[]
                for name,minimum,maximum in [('title',5,100),('problem',10,800),('solution',10,1200),('audience',3,300)]:
                    value=body.get(name)
                    if not isinstance(value,str) or not minimum<=len(value.strip())<=maximum: raise Failure(400,'input')
                    fields.append(value.strip())
                existing=db.execute('SELECT id FROM ideas WHERE author=? AND title=? AND problem=? AND solution=? AND audience=?',(user['id'],*fields)).fetchone()
                if existing:
                    result['id']=existing[0]
                    db.execute('INSERT INTO requests VALUES(?,?,?,?)',(user['id'],key,digest,json.dumps(result)))
                    return result
                if db.execute('SELECT count(*) FROM ideas WHERE author=?',(user['id'],)).fetchone()[0]>=3: raise Failure(429,'quota')
                result['id']=db.execute('INSERT INTO ideas(author,title,problem,solution,audience) VALUES(?,?,?,?,?)',(user['id'],*fields)).lastrowid
            elif action=='vote':
                if r['phase'] not in ('voting','runoff'): raise Failure(409,'phase')
                idea=body.get('idea')
                if type(idea) is not int or not db.execute("SELECT 1 FROM ideas WHERE id=? AND status='approved'",(idea,)).fetchone(): raise Failure(400,'input')
                if r['ballot']!='main' and not db.execute('SELECT 1 FROM candidates WHERE ballot=? AND idea=?',(r['ballot'],idea)).fetchone(): raise Failure(400,'input')
                db.execute('INSERT INTO votes VALUES(?,?,?) ON CONFLICT(account,ballot) DO UPDATE SET idea=excluded.idea',(user['id'],r['ballot'],idea))
            elif action=='review':
                if user['role']!='admin': raise Failure(403,'forbidden')
                if r['phase'] not in ('submissions','draft'): raise Failure(409,'phase')
                status=body.get('status'); reason=body.get('reason','')
                if not isinstance(reason,str): raise Failure(400,'input')
                reason=reason.strip()
                if status not in ('approved','rejected') or (status=='rejected' and not 5<=len(reason)<=300): raise Failure(400,'input')
                if db.execute("UPDATE ideas SET status=?,reason=? WHERE id=? AND status='pending'",(status,reason,body.get('idea'))).rowcount!=1: raise Failure(409,'input')
                db.execute('INSERT INTO audit(account,action,idea,reason,at) VALUES(?,?,?,?,?)',(user['id'],status,body.get('idea'),reason,now))
            elif action=='runoff':
                if user['role']!='admin': raise Failure(403,'forbidden')
                if r['phase']!='runoff_pending': raise Failure(409,'phase')
                deadline=body.get('deadline')
                if type(deadline) is not int or deadline<=now+60: raise Failure(400,'dates')
                db.execute("UPDATE round SET phase='runoff',runoff_closes=?",(deadline,))
                db.execute('INSERT INTO audit(account,action,reason,at) VALUES(?,?,?,?)',(user['id'],'runoff',str(deadline),now))
            else: raise Failure(404,'input')
            db.execute('INSERT INTO requests VALUES(?,?,?,?)',(user['id'],key,digest,json.dumps(result)))
            return result

class Handler(SimpleHTTPRequestHandler):
    store=None
    def __init__(self,*args,**kwargs): super().__init__(*args,directory=str(ROOT),**kwargs)
    def log_message(self,*args): pass  # Do not log codes, cookies or submission content.
    def end_headers(self):
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Cache-Control','no-store')
        super().end_headers()
    def identity(self):
        cookie=SimpleCookie()
        try: cookie.load(self.headers.get('Cookie',''))
        except Exception: return None
        return self.store.identity(cookie['contest_session'].value,int(time.time())) if 'contest_session' in cookie else None
    def send_json(self,status,body,cookie=None):
        payload=json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8')
        if cookie: self.send_header('Set-Cookie',cookie)
        self.send_header('Content-Length',str(len(payload))); self.end_headers(); self.wfile.write(payload)
    def do_GET(self):
        if self.path.split('?')[0]=='/api/contest':
            user=self.identity(); data=self.store.state(user,int(time.time()))
            data['csrf']=user['csrf'] if user else None
            return self.send_json(200,data)
        # Serve public assets only; never expose databases, source or dotfiles.
        path=self.path.split('?')[0]
        if not (path=='/' or (path.count('/')==1 and path.endswith('.html') and (ROOT/path[1:]).is_file()) or (path.startswith('/assets/') and Path(path).suffix in ('.js','.css','.svg','.png','.json'))): return self.send_error(404)
        return super().do_GET()
    def do_POST(self):
        now=int(time.time())
        try:
            expected='http://127.0.0.1:'+str(self.server.server_port)
            if self.headers.get('Origin')!=expected or self.headers.get('Host')!='127.0.0.1:'+str(self.server.server_port): raise Failure(403,'origin')
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=12000 or 'application/json' not in self.headers.get('Content-Type',''): raise Failure(400,'input')
            body=json.loads(self.rfile.read(size))
            if not isinstance(body,dict): raise Failure(400,'input')
            action=self.path.removeprefix('/api/')
            if action=='login':
                self.store.limit(self.client_address[0],'login',6,now)
                token,_=self.store.login(body.get('code',''),now)
                return self.send_json(200,{'ok':True},f'contest_session={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=3600')
            user=self.identity()
            if not user: raise Failure(401,'auth')
            if not hmac.compare_digest(self.headers.get('X-CSRF-Token',''),user['csrf']): raise Failure(403,'csrf')
            if action=='logout':
                with self.store.db() as db: db.execute('DELETE FROM sessions WHERE token=?',(hashlib.sha256(SimpleCookie(self.headers['Cookie'])['contest_session'].value.encode()).hexdigest(),))
                return self.send_json(200,{'ok':True},'contest_session=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0')
            self.store.limit(user['id'],action,5 if action=='submit' else 30,now)
            return self.send_json(200,self.store.act(user,action,body,self.headers.get('Idempotency-Key'),now))
        except Failure as error: self.send_json(error.status,{'error':error.code})
        except (ValueError,TypeError,KeyError): self.send_json(400,{'error':'input'})
        except sqlite3.Error: self.send_json(503,{'error':'unavailable'})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8771)
    parser.add_argument('--init-demo',action='store_true')
    parser.add_argument('--voting-demo',action='store_true')
    args=parser.parse_args()
    if args.db.resolve().is_relative_to(ROOT):
        raise SystemExit('Keep the database outside the public checkout directory.')
    args.db.parent.mkdir(parents=True,exist_ok=True)
    store=Store(args.db)
    if args.init_demo:
        print('LOCAL DEMO ONLY — keep codes private; no Telegram identity or real prize.')
        print('Owner demo code:',store.create_account('admin'))
        print('Member demo code:',store.create_account())
        now=int(time.time()); store.schedule(now-120,now+86400,now+172800,now+259200)
        if args.voting_demo:
            # Explicit demonstration entries, never represented as community submissions.
            with store.db() as db:
                for title in ('DEMO · Clearer data dates','DEMO · Ratio explanation'):
                    db.execute("INSERT INTO ideas(author,title,problem,solution,audience,status) VALUES(1,?,?,?,?, 'approved')",(title,'DEMO: help readers understand the data.','DEMO: explain source dates and units clearly.','DEMO visitors'))
                db.execute("UPDATE round SET phase='voting',submits=?",(now-60,))
    Handler.store=store
    print(f'Preview: http://127.0.0.1:{args.port}/contest.html',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

if __name__=='__main__': main()
