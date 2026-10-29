function decodeBase64Url(value) {
  const normalized = value.replaceAll('-', '+').replaceAll('_', '/');
  const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, '=');
  return Uint8Array.from(atob(padded), character => character.charCodeAt(0));
}

function encodeBase64Url(value) {
  const bytes = new Uint8Array(value);
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll('+', '-').replaceAll('/', '_').replaceAll('=', '');
}

function ensureWebAuthn() {
  if (!window.isSecureContext || !window.PublicKeyCredential || !navigator.credentials) {
    throw new Error('Passkeys are not supported in this browser or secure context.');
  }
}

export async function createPasskey(options) {
  ensureWebAuthn();
  const publicKey = {
    ...options,
    challenge: decodeBase64Url(options.challenge),
    user: { ...options.user, id: decodeBase64Url(options.user.id) },
    excludeCredentials: (options.excludeCredentials || []).map(item => ({
      ...item, id: decodeBase64Url(item.id)
    }))
  };
  let credential;
  try {
    credential = await navigator.credentials.create({ publicKey });
  } catch (error) {
    if (error?.name === 'NotAllowedError') throw new Error('Passkey setup was cancelled or timed out.');
    throw error;
  }
  if (!credential) throw new Error('The browser did not create a passkey.');
  return {
    id: credential.id,
    rawId: encodeBase64Url(credential.rawId),
    type: credential.type,
    authenticatorAttachment: credential.authenticatorAttachment || undefined,
    clientExtensionResults: credential.getClientExtensionResults(),
    response: {
      clientDataJSON: encodeBase64Url(credential.response.clientDataJSON),
      attestationObject: encodeBase64Url(credential.response.attestationObject),
      transports: credential.response.getTransports?.() || []
    }
  };
}

export async function authenticatePasskey(options) {
  ensureWebAuthn();
  const publicKey = {
    ...options,
    challenge: decodeBase64Url(options.challenge),
    allowCredentials: (options.allowCredentials || []).map(item => ({
      ...item, id: decodeBase64Url(item.id)
    }))
  };
  let credential;
  try {
    credential = await navigator.credentials.get({ publicKey });
  } catch (error) {
    if (error?.name === 'NotAllowedError') throw new Error('Passkey verification was cancelled or timed out.');
    throw error;
  }
  if (!credential) throw new Error('The browser did not return a passkey.');
  return {
    id: credential.id,
    rawId: encodeBase64Url(credential.rawId),
    type: credential.type,
    authenticatorAttachment: credential.authenticatorAttachment || undefined,
    clientExtensionResults: credential.getClientExtensionResults(),
    response: {
      clientDataJSON: encodeBase64Url(credential.response.clientDataJSON),
      authenticatorData: encodeBase64Url(credential.response.authenticatorData),
      signature: encodeBase64Url(credential.response.signature),
      userHandle: credential.response.userHandle ? encodeBase64Url(credential.response.userHandle) : undefined
    }
  };
}
