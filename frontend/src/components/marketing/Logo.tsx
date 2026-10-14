import Link from 'next/link';
import { Zap } from 'lucide-react';

export function Logo({ dark = false }: { dark?: boolean }) {
  return (
    <Link href="/" className="flex items-center gap-2" aria-label="EnMS home">
      <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-brand-500 to-brand-700 shadow-sm shadow-brand-700/30">
        <Zap className="h-4 w-4 text-white" />
      </span>
      <span className={`text-lg font-bold tracking-tight ${dark ? 'text-white' : 'text-ink-950'}`}>EnMS</span>
    </Link>
  );
}
