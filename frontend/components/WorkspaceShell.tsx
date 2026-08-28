import Link from "next/link";
import type { ReactNode } from "react";
import { ClerkWorkspaceControls } from "./ClerkControls";
import { isClerkFrontendEnabled } from "../lib/auth-config";
import { loadWorkspaceContext } from "../lib/workspace-context";

const navItems = [
  { href: "/dashboard", label: "Analysis" },
  { href: "/uploads", label: "Source files" },
  { href: "/runs", label: "Run history" },
  { href: "/reports", label: "Reports" },
  { href: "/settings", label: "Setup" }
];

type WorkspaceShellProps = {
  currentPath: string;
  title: string;
  eyebrow?: string;
  modeLabel: string;
  modeDescription: string;
  modeTone?: "info" | "error";
  children: ReactNode;
};

export async function WorkspaceShell({
  currentPath,
  title,
  eyebrow = "Workspace",
  modeLabel,
  modeDescription,
  modeTone = "info",
  children
}: WorkspaceShellProps) {
  const workspace = await loadWorkspaceContext();
  const useClerkControls = isClerkFrontendEnabled() && workspace.provider === "clerk";

  return (
    <main className="appShell">
      <aside className="sidebar">
        <Link href="/" className="sidebarBrandRow" aria-label="RDS Energy return to home">
          <span className="sidebarBrandMark" aria-hidden />
          <div className="sidebarBrandMeta">
            <strong>RDS Energy</strong>
            <span className="sidebarTagline">Baseline analytics</span>
          </div>
        </Link>
        <div className="tenantBlock">
          <span className="tenantLabel">Organization</span>
          <strong>{workspace.orgName}</strong>
          <span className="tenantMeta">{workspace.orgSlug}</span>
        </div>
        <div className="tenantActions">
          <span className="roleBadge">{workspace.roleLabel}</span>
          {useClerkControls ? (
            <ClerkWorkspaceControls enabled={useClerkControls} />
          ) : (
            <Link href="/organizations" className="tenantLink">
              Switch
            </Link>
          )}
        </div>
        <nav className="sideNav" aria-label="Workspace sections">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={item.href === currentPath ? "navActive" : undefined}
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </aside>

      <section className="workspace">
        <header className="workspaceHeader">
          <div>
            <p className="eyebrow">{eyebrow}</p>
            <h1>{title}</h1>
          </div>
          {useClerkControls ? null : (
            <Link href={workspace.hasLiveSession ? "/logout" : "/login"} className="userMenu">
              Sign out
            </Link>
          )}
        </header>

        <div
          className={modeTone === "error" ? "modeBanner modeBannerError" : "modeBanner"}
          role={modeTone === "error" ? "alert" : "status"}
          aria-live={modeTone === "error" ? "assertive" : "polite"}
        >
          <strong>{modeLabel}</strong>
          <span>{modeDescription}</span>
        </div>

        {children}
      </section>
    </main>
  );
}
