import Link from 'next/link';
import { Zap } from 'lucide-react';

export default function Logo() {
  return (
    <Link href="/" className="flex items-center gap-2">
      <span className="flex h-8 w-8 items-center justify-center rounded-sm bg-brand-400">
        <Zap className="h-4 w-4 text-ink-950" />
      </span>
      <span className="text-lg font-semibold tracking-tight text-white">EnMS</span>
    </Link>
  );
}
