import Link from "next/link";
import type { ReactNode } from "react";
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
  children: ReactNode;
};

export async function WorkspaceShell({
  currentPath,
  title,
  eyebrow = "Workspace",
  modeLabel,
  modeDescription,
  children
}: WorkspaceShellProps) {
  const workspace = await loadWorkspaceContext();

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
          <Link href="/organizations" className="tenantLink">
            Switch
          </Link>
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
          <Link href={workspace.hasLiveSession ? "/logout" : "/login"} className="userMenu">
            Sign out
          </Link>
        </header>

        <div className="modeBanner" role="status" aria-live="polite">
          <strong>{modeLabel}</strong>
          <span>{modeDescription}</span>
        </div>

        {children}
      </section>
    </main>
  );
}
