import { UserManager, type User, type UserManagerSettings } from "oidc-client-ts";

const domain = import.meta.env.VITE_COGNITO_DOMAIN as string | undefined;
const clientId = import.meta.env.VITE_COGNITO_CLIENT_ID as string | undefined;
const issuer = import.meta.env.VITE_COGNITO_ISSUER as string | undefined;
const redirectUri = (import.meta.env.VITE_COGNITO_REDIRECT_URI as string | undefined) ?? `${window.location.origin}/auth/callback`;

export const authEnabled = Boolean(domain && clientId && issuer);
const settings: UserManagerSettings | null = authEnabled ? {
  authority: domain!, client_id: clientId!, redirect_uri: redirectUri,
  post_logout_redirect_uri: window.location.origin, response_type: "code", scope: "openid email profile",
  automaticSilentRenew: false,
  metadata: { issuer: issuer!, authorization_endpoint: `${domain}/oauth2/authorize`, token_endpoint: `${domain}/oauth2/token`, userinfo_endpoint: `${domain}/oauth2/userInfo`, end_session_endpoint: `${domain}/logout`, jwks_uri: `${issuer}/.well-known/jwks.json` },
} : null;

export const userManager = settings ? new UserManager(settings) : null;
export function currentUser(): Promise<User | null> { return userManager?.getUser() ?? Promise.resolve(null); }
export function signIn(): Promise<void> { return userManager ? userManager.signinRedirect() : Promise.resolve(); }
export async function finishCallback(): Promise<User | null> { return userManager ? ((await userManager.signinCallback()) ?? null) : null; }
export async function signOut(): Promise<void> { if (!userManager) return; await userManager.removeUser(); window.location.assign(`${domain}/logout?client_id=${encodeURIComponent(clientId!)}&logout_uri=${encodeURIComponent(window.location.origin)}`); }
export async function accessToken(): Promise<string | undefined> { const user = await currentUser(); return user && !user.expired ? user.access_token : undefined; }
export async function isAdmin(): Promise<boolean> { const user = await currentUser(); const groups = user?.profile["cognito:groups"]; return Array.isArray(groups) && groups.includes("target-prioritization-admin"); }
