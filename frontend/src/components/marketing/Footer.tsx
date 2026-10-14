import Link from 'next/link';
import { Logo } from './Logo';

const columns = [
  {
    title: 'Product',
    links: [
      { href: '/#how-it-works', label: 'How it works' },
      { href: '/#features', label: 'Features' },
      { href: '/pricing', label: 'Pricing' },
      { href: '/demo', label: 'Live demo' },
    ],
  },
  {
    title: 'ISO 50001',
    links: [
      { href: '/iso-50001', label: 'What the standard requires' },
      { href: '/iso-50001#clauses', label: 'Clause 4–10 mapping' },
      { href: '/#readiness', label: 'Readiness score' },
      { href: '/#faq', label: 'FAQ' },
    ],
  },
  {
    title: 'Account',
    links: [
      { href: '/signup', label: 'Start free trial' },
      { href: '/login', label: 'Log in' },
    ],
  },
];

export function MarketingFooter() {
  return (
    <footer className="border-t border-ink-100 bg-white">
      <div className="mx-auto grid max-w-7xl gap-10 px-6 py-14 md:grid-cols-5">
        <div className="md:col-span-2">
          <Logo />
          <p className="mt-4 max-w-xs text-sm leading-relaxed text-ink-500">
            The energy management system software for teams pursuing ISO 50001:2018 certification —
            weather-normalised baselines, SEUs, EnPIs and audit-ready evidence in one workspace.
          </p>
        </div>
        {columns.map((c) => (
          <div key={c.title}>
            <h4 className="text-sm font-semibold text-ink-950">{c.title}</h4>
            <ul className="mt-4 space-y-2.5">
              {c.links.map((l) => (
                <li key={l.href}>
                  <Link href={l.href} className="text-sm text-ink-500 hover:text-ink-950">{l.label}</Link>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <div className="border-t border-ink-100">
        <div className="mx-auto flex max-w-7xl flex-col items-start justify-between gap-2 px-6 py-6 text-xs text-ink-400 sm:flex-row sm:items-center">
          <p>&copy; {new Date().getFullYear()} EnMS. All rights reserved.</p>
          <p>ISO 50001 is a standard published by the International Organization for Standardization. EnMS is not affiliated with ISO.</p>
        </div>
      </div>
    </footer>
  );
}
