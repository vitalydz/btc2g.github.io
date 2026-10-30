// Local software authenticator. Never replaces the production verification functions.
import { generateKeyPairSync, randomBytes, createHash, sign } from 'node:crypto';
const hash = x => createHash('sha256').update(x).digest();
const b64 = x => Buffer.from(x).toString('base64url');
function cbor(x) {
  const head = (major, n) => n < 24 ? Buffer.from([major * 32 + n]) :
    n < 256 ? Buffer.from([major * 32 + 24, n]) : Buffer.from([major * 32 + 25, n >> 8, n & 255]);
  if (Number.isInteger(x)) return x >= 0 ? head(0,x) : head(1,-1-x);
  if (Buffer.isBuffer(x)) return Buffer.concat([head(2,x.length),x]);
  if (typeof x === 'string') { const b=Buffer.from(x); return Buffer.concat([head(3,b.length),b]); }
  const entries = x instanceof Map ? [...x] : Object.entries(x);
  return Buffer.concat([head(5,entries.length), ...entries.flatMap(([k,v])=>[cbor(k),cbor(v)])]);
}
export function authenticator() {
  const {privateKey,publicKey}=generateKeyPairSync('ec',{namedCurve:'prime256v1'});
  const jwk=publicKey.export({format:'jwk'}), id=randomBytes(32);
  const cose=cbor(new Map([[1,2],[3,-7],[-1,1],[-2,Buffer.from(jwk.x,'base64url')],[-3,Buffer.from(jwk.y,'base64url')]]));
  const client=(type,challenge,origin)=>Buffer.from(JSON.stringify({type,challenge,origin,crossOrigin:false}));
  const auth=(rp,flags,counter)=>{const count=Buffer.alloc(4);count.writeUInt32BE(counter);return Buffer.concat([hash(rp),Buffer.from([flags]),count]);};
  return {
    registration(options,origin='https://metrics.test') {
      const length=Buffer.alloc(2);length.writeUInt16BE(id.length);
      const data=Buffer.concat([auth(options.rp.id,0x45,0),Buffer.alloc(16),length,id,cose]);
      return {id:b64(id),rawId:b64(id),type:'public-key',clientExtensionResults:{},response:{
        clientDataJSON:b64(client('webauthn.create',options.challenge,origin)),
        attestationObject:b64(cbor({fmt:'none',attStmt:{},authData:data})),transports:['internal']}};
    },
    assertion(options,{origin='https://metrics.test',counter=1,flags=5,badSignature=false,rp=options.rpId}={}) {
      const data=auth(rp,flags,counter), json=client('webauthn.get',options.challenge,origin);
      const signature=sign('sha256',Buffer.concat([data,hash(json)]),privateKey);
      if(badSignature) signature[signature.length-1]^=1;
      return {id:b64(id),rawId:b64(id),type:'public-key',clientExtensionResults:{},response:{
        authenticatorData:b64(data),clientDataJSON:b64(json),signature:b64(signature),userHandle:null}};
    }
  };
}
