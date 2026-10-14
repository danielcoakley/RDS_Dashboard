'use client';

import Link from 'next/link';
import { useState } from 'react';
import { Menu, X } from 'lucide-react';
import { useAuth } from '@/lib/auth';
import { Logo } from './Logo';

const links = [
  { href: '/#how-it-works', label: 'How it works' },
  { href: '/#features', label: 'Features' },
  { href: '/iso-50001', label: 'ISO 50001' },
  { href: '/pricing', label: 'Pricing' },
  { href: '/#faq', label: 'FAQ' },
];

export function MarketingNav() {
  const { user } = useAuth();
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 border-b border-ink-100/80 bg-white/80 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
        <Logo />
        <nav className="hidden items-center gap-8 md:flex">
          {links.map((l) => (
            <Link key={l.href} href={l.href} className="text-sm font-medium text-ink-600 transition hover:text-ink-950">
              {l.label}
            </Link>
          ))}
        </nav>
        <div className="hidden items-center gap-3 md:flex">
          {user ? (
            <Link href="/dashboard" className="rounded-lg bg-ink-950 px-4 py-2 text-sm font-semibold text-white hover:bg-ink-800">
              Open dashboard
            </Link>
          ) : (
            <>
              <Link href="/login" className="px-3 py-2 text-sm font-medium text-ink-600 hover:text-ink-950">Log in</Link>
              <Link href="/demo" className="rounded-lg border border-ink-200 px-4 py-2 text-sm font-semibold text-ink-800 hover:border-ink-300 hover:bg-ink-50">
                Live demo
              </Link>
              <Link href="/signup" className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white shadow-sm shadow-brand-600/30 hover:bg-brand-700">
                Start free trial
              </Link>
            </>
          )}
        </div>
        <button
          type="button"
          aria-label={open ? 'Close menu' : 'Open menu'}
          onClick={() => setOpen((o) => !o)}
          className="rounded-lg p-2 text-ink-700 hover:bg-ink-50 md:hidden"
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>
      {open && (
        <div className="border-t border-ink-100 bg-white px-6 py-4 md:hidden">
          <div className="flex flex-col gap-3">
            {links.map((l) => (
              <Link key={l.href} href={l.href} onClick={() => setOpen(false)} className="text-sm font-medium text-ink-700">
                {l.label}
              </Link>
            ))}
            <div className="mt-2 grid grid-cols-2 gap-2">
              {user ? (
                <Link href="/dashboard" className="col-span-2 rounded-lg bg-ink-950 px-4 py-2.5 text-center text-sm font-semibold text-white">
                  Open dashboard
                </Link>
              ) : (
                <>
                  <Link href="/login" className="rounded-lg border border-ink-200 px-4 py-2.5 text-center text-sm font-semibold text-ink-800">Log in</Link>
                  <Link href="/signup" className="rounded-lg bg-brand-600 px-4 py-2.5 text-center text-sm font-semibold text-white">Start free trial</Link>
                  <Link href="/demo" className="col-span-2 rounded-lg border border-ink-200 px-4 py-2.5 text-center text-sm font-semibold text-ink-800">Explore live demo</Link>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
