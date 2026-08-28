import { cookies } from "next/headers";
import { auth } from "@clerk/nextjs/server";
import { isClerkFrontendEnabled } from "./auth-config";

export const SESSION_COOKIE_NAME = "rds_auth_token";
export const USER_ID_COOKIE_NAME = "rds_user_id";
export const ORG_ID_COOKIE_NAME = "rds_org_id";
export const ROLE_COOKIE_NAME = "rds_role";

export type AppSession = {
  userId: string | null;
  organizationId: string | null;
  organizationSlug?: string | null;
  role: string | null;
  authToken: string | null;
  provider?: "clerk" | "dev" | null;
};

function normalizeClerkRole(role: string | null | undefined): string | null {
  if (!role) {
    return null;
  }
  const normalized = role.toLowerCase().replace(/^org:/, "");
  if (normalized === "admin") {
    return "owner";
  }
  if (normalized === "member" || normalized === "basic_member") {
    return "viewer";
  }
  return normalized;
}

async function readClerkSession(): Promise<AppSession | null> {
  if (!isClerkFrontendEnabled()) {
    return null;
  }

  const clerkSession = await auth();
  if (!clerkSession.userId) {
    return null;
  }

  return {
    userId: clerkSession.userId,
    organizationId: clerkSession.orgId ?? null,
    organizationSlug: clerkSession.orgSlug ?? null,
    role: normalizeClerkRole(clerkSession.orgRole),
    authToken: await clerkSession.getToken(),
    provider: "clerk"
  };
}

export async function readAppSession(): Promise<AppSession> {
  const clerkSession = await readClerkSession();
  if (clerkSession) {
    return clerkSession;
  }

  const cookieStore = await cookies();
  return {
    userId: cookieStore.get(USER_ID_COOKIE_NAME)?.value ?? null,
    organizationId: cookieStore.get(ORG_ID_COOKIE_NAME)?.value ?? null,
    organizationSlug: null,
    role: cookieStore.get(ROLE_COOKIE_NAME)?.value ?? null,
    authToken: cookieStore.get(SESSION_COOKIE_NAME)?.value ?? null,
    provider: cookieStore.get(USER_ID_COOKIE_NAME)?.value ? "dev" : null
  };
}

export async function writeAppSession(session: AppSession): Promise<void> {
  const cookieStore = await cookies();
  const maxAge = 60 * 60 * 12;
  const options = {
    httpOnly: true,
    sameSite: "lax" as const,
    secure: false,
    path: "/",
    maxAge
  };

  if (session.authToken) {
    cookieStore.set(SESSION_COOKIE_NAME, session.authToken, options);
  }
  if (session.userId) {
    cookieStore.set(USER_ID_COOKIE_NAME, session.userId, options);
  }
  if (session.organizationId) {
    cookieStore.set(ORG_ID_COOKIE_NAME, session.organizationId, options);
  } else {
    cookieStore.delete(ORG_ID_COOKIE_NAME);
  }
  if (session.role) {
    cookieStore.set(ROLE_COOKIE_NAME, session.role, options);
  } else {
    cookieStore.delete(ROLE_COOKIE_NAME);
  }
}

export async function clearAppSession(): Promise<void> {
  const cookieStore = await cookies();
  cookieStore.delete(SESSION_COOKIE_NAME);
  cookieStore.delete(USER_ID_COOKIE_NAME);
  cookieStore.delete(ORG_ID_COOKIE_NAME);
  cookieStore.delete(ROLE_COOKIE_NAME);
}
