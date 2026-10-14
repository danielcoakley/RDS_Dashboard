import Link from 'next/link';
import { Check } from 'lucide-react';
import Nav from '@/components/marketing/Nav';
import Footer from '@/components/marketing/Footer';
import { MarketingShell, SectionHeading } from '@/components/marketing/ui';
import FAQ from '@/components/marketing/FAQ';

export const metadata = { title: 'Pricing — EnMS' };

const tiers = [
  { name: 'Starter', price: '€149', per: '/mo', sites: '1 site', cta: 'Start free trial', href: '/signup',
    items: ['Weather-normalised baseline (HDD/CDD)', 'Automatic SEU identification', 'EnPI tracking vs baseline', 'Clause 4–10 readiness score', 'Evidence export'] },
  { name: 'Professional', price: '€449', per: '/mo', sites: 'Up to 10 sites', cta: 'Start free trial', href: '/signup', featured: true,
    items: ['Everything in Starter', 'Multi-site normalisation', 'Portfolio EnPI comparison', 'Objectives & action plans', 'Management review packs'] },
  { name: 'Enterprise', price: 'Custom', per: '', sites: 'Unlimited sites', cta: 'Contact sales', href: 'mailto:sales@enms.app',
    items: ['Everything in Professional', 'Group-level certification support', 'SSO & advanced permissions', 'Dedicated onboarding', 'Priority support & SLA'] },
];

export default function PricingPage() {
  return (
    <MarketingShell>
      <Nav />
      <main className="mx-auto max-w-7xl px-6 py-20 lg:py-28">
        <div className="text-center [&>div]:mx-auto">
          <SectionHeading eyebrow="Pricing" title="Transparent pricing for every EnMS." sub="Every plan includes a 14-day free trial. No credit card required." />
        </div>
        <div className="mt-16 grid gap-6 lg:grid-cols-3">
          {tiers.map((t) => (
            <div key={t.name} className={t.featured ? 'mk-conic rounded-lg p-px' : 'rounded-lg border border-white/10'}>
              <div className="flex h-full flex-col rounded-lg bg-ink-900 p-8">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-white">{t.name}</h3>
                  {t.featured && <span className="rounded-full bg-brand-400/10 px-2.5 py-0.5 font-mono text-[11px] text-brand-400">Most popular</span>}
                </div>
                <p className="mt-1 font-mono text-xs text-ink-400">{t.sites}</p>
                <p className="mt-6"><span className="font-mono text-4xl text-white">{t.price}</span><span className="text-ink-400">{t.per}</span></p>
                <ul className="mt-8 flex-1 space-y-3">
                  {t.items.map((i) => (
                    <li key={i} className="flex gap-3 text-sm text-ink-200"><Check className="h-4 w-4 shrink-0 text-brand-400" />{i}</li>
                  ))}
                </ul>
                <Link href={t.href} className={`mt-8 rounded-sm px-4 py-3 text-center font-semibold ${t.featured ? 'bg-brand-400 text-ink-950 hover:bg-brand-300' : 'border border-white/15 text-white hover:bg-white/5'}`}>{t.cta}</Link>
              </div>
            </div>
          ))}
        </div>
        <p className="mt-6 text-center text-xs text-ink-400">Prices exclude VAT. Billed monthly.</p>
        <div className="mx-auto mt-24 max-w-3xl"><FAQ /></div>
      </main>
      <Footer />
    </MarketingShell>
  );
}
