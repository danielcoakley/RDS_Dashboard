import Link from 'next/link';
import Logo from './Logo';

const cols = [
  { title: 'Product', links: [['How it works', '/#how'], ['Features', '/#features'], ['Pricing', '/pricing'], ['Live demo', '/demo']] },
  { title: 'Compliance', links: [['ISO 50001:2018', '/iso-50001'], ['Clause readiness', '/#readiness'], ['FAQ', '/#faq']] },
  { title: 'Account', links: [['Log in', '/login'], ['Start free trial', '/signup']] },
];

export default function Footer() {
  return (
    <footer className="border-t border-white/10 bg-ink-900">
      <div className="mx-auto grid max-w-7xl gap-10 px-6 py-16 sm:grid-cols-2 lg:grid-cols-4">
        <div>
          <Logo />
          <p className="mt-4 max-w-xs text-sm leading-relaxed text-ink-300">
            The energy management system built for ISO 50001:2018 certification.
          </p>
        </div>
        {cols.map((c) => (
          <div key={c.title}>
            <h4 className="font-mono text-xs uppercase tracking-widest text-ink-400">{c.title}</h4>
            <ul className="mt-4 space-y-3">
              {c.links.map(([label, href]) => (
                <li key={label}><Link href={href} className="text-sm text-ink-200 hover:text-white">{label}</Link></li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <div className="border-t border-white/10">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-6 py-6 text-xs text-ink-400 sm:flex-row sm:items-center sm:justify-between">
          <span className="flex items-center gap-2 font-mono">
            <span className="h-2 w-2 animate-pulse rounded-full bg-brand-400 shadow-[0_0_8px] shadow-brand-400" />
            System status: All systems operational
          </span>
          <span>© {new Date().getFullYear()} EnMS. All rights reserved.</span>
        </div>
      </div>
    </footer>
  );
}
