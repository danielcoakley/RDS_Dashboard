import Link from "next/link";
import type { ReactNode } from "react";

const navItems = [
  { href: "/dashboard", label: "Overview" },
  { href: "/uploads", label: "Uploads" },
  { href: "/runs", label: "Runs" },
  { href: "/reports", label: "Reports" },
  { href: "/settings", label: "Settings" }
];

type WorkspaceShellProps = {
  currentPath: string;
  title: string;
  eyebrow?: string;
  modeLabel: string;
  modeDescription: string;
  children: ReactNode;
};

export function WorkspaceShell({
  currentPath,
  title,
  eyebrow = "Tenant dashboard",
  modeLabel,
  modeDescription,
  children
}: WorkspaceShellProps) {
  return (
    <main className="appShell">
      <aside className="sidebar">
        <Link href="/" className="sidebarBrandRow" aria-label="RDS Energy return to home">
          <span className="sidebarBrandMark" aria-hidden />
          <div className="sidebarBrandMeta">
            <strong>RDS Energy</strong>
            <span className="sidebarTagline">SaaS workspace</span>
          </div>
        </Link>
        <div className="tenantBlock">
          <span className="tenantLabel">Tenant</span>
          <strong>RDS Site</strong>
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
          <div className="userMenu">Owner</div>
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
