'use client';

import Link from 'next/link';
import { useState } from 'react';
import { Menu, X } from 'lucide-react';
import { useAuth } from '@/lib/auth';
import Logo from './Logo';

const links = [
  { href: '/#how', label: 'How it works' },
  { href: '/iso-50001', label: 'ISO 50001' },
  { href: '/pricing', label: 'Pricing' },
  { href: '/#faq', label: 'FAQ' },
];

export default function Nav() {
  const { user } = useAuth();
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-ink-950/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
        <Logo />
        <nav className="hidden items-center gap-8 md:flex">
          {links.map((l) => (
            <Link key={l.href} href={l.href} className="text-sm text-ink-300 transition hover:text-white">{l.label}</Link>
          ))}
        </nav>
        <div className="hidden items-center gap-3 md:flex">
          {user ? (
            <Link href="/dashboard" className="rounded-sm bg-brand-400 px-4 py-2 text-sm font-semibold text-ink-950 hover:bg-brand-300">Dashboard</Link>
          ) : (
            <>
              <Link href="/login" className="rounded-sm border border-white/15 px-4 py-2 text-sm font-medium text-ink-100 hover:border-white/30 hover:text-white">Log in</Link>
              <Link href="/signup" className="rounded-sm bg-brand-400 px-4 py-2 text-sm font-semibold text-ink-950 hover:bg-brand-300">Start free trial</Link>
            </>
          )}
        </div>
        <button aria-label="Toggle menu" onClick={() => setOpen(!open)} className="text-ink-200 md:hidden">
          {open ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </button>
      </div>
      {open && (
        <div className="border-t border-white/10 px-6 py-4 md:hidden">
          <div className="flex flex-col gap-4">
            {links.map((l) => (
              <Link key={l.href} href={l.href} onClick={() => setOpen(false)} className="text-ink-200">{l.label}</Link>
            ))}
            <Link href="/login" className="rounded-sm border border-white/15 px-4 py-2 text-center text-ink-100">Log in</Link>
            <Link href="/signup" className="rounded-sm bg-brand-400 px-4 py-2 text-center font-semibold text-ink-950">Start free trial</Link>
          </div>
        </div>
      )}
    </header>
  );
}
