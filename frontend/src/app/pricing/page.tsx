import type { Metadata } from 'next';
import Link from 'next/link';
import { Check } from 'lucide-react';
import { MarketingNav } from '@/components/marketing/Nav';
import { MarketingFooter } from '@/components/marketing/Footer';
import { FinalCta } from '@/components/marketing/CtaSection';

export const metadata: Metadata = {
  title: 'Pricing — EnMS ISO 50001 energy management software',
  description:
    'Simple per-site pricing for ISO 50001 energy management: Starter €149/month, Professional €449/month for up to 10 sites, and Enterprise plans for portfolios. 14-day free trial.',
};

const tiers = [
  {
    name: 'Starter',
    price: '€149',
    period: '/month',
    blurb: 'For a single site preparing for its first ISO 50001 certification.',
    cta: { label: 'Start free trial', href: '/signup' },
    highlight: false,
    features: [
      '1 site',
      'Weather-normalised baseline (HDD/CDD)',
      'SEU identification & EnPI tracking',
      'Clause 4–10 readiness score',
      'PDF & CSV report exports',
      'Up to 3 users',
    ],
  },
  {
    name: 'Professional',
    price: '€449',
    period: '/month',
    blurb: 'For multi-site organisations running a certified EnMS.',
    cta: { label: 'Start free trial', href: '/signup' },
    highlight: true,
    features: [
      'Up to 10 sites',
      'Everything in Starter',
      'Objectives, targets & action plans',
      'Internal audit & nonconformity tracking',
      'Portfolio-level EnPIs',
      'Unlimited users with Owner / Manager / Viewer roles',
      'Audit log',
    ],
  },
  {
    name: 'Enterprise',
    price: 'Custom',
    period: '',
    blurb: 'For large portfolios, consultancies and multi-entity groups.',
    cta: { label: 'Explore live demo', href: '/demo' },
    highlight: false,
    features: [
      'Unlimited sites',
      'Everything in Professional',
      'SSO & advanced access controls',
      'Data residency options',
      'Onboarding & baseline review support',
      'Dedicated success manager & SLA',
    ],
  },
];

const faqs = [
  { q: 'What counts as a site?', a: 'A site is a physical location within your EnMS scope and boundaries, such as a plant, building or campus, regardless of how many meters it has.' },
  { q: 'Is there a free trial?', a: 'Yes. Every plan starts with a 14-day free trial with full functionality. No credit card is required to start.' },
  { q: 'Can I change plans later?', a: 'You can upgrade or downgrade at any time. Changes are prorated on your next invoice.' },
  { q: 'Are prices inclusive of VAT?', a: 'Prices are shown excluding VAT. Annual billing is available on request.' },
];

export default function PricingPage() {
  return (
    <div className="min-h-screen bg-white">
      <MarketingNav />

      <section className="relative overflow-hidden">
        <div className="pointer-events-none absolute inset-0 marketing-grid" />
        <div className="relative mx-auto max-w-3xl px-6 pb-12 pt-20 text-center">
          <p className="text-sm font-semibold uppercase tracking-wider text-brand-600">Pricing</p>
          <h1 className="mt-3 text-4xl font-bold tracking-tight text-ink-950 sm:text-5xl">
            Less than a consultant day per month
          </h1>
          <p className="mt-5 text-lg text-ink-600">
            Straightforward per-site pricing. Every plan includes weather-normalised baselines, EnPIs and the ISO 50001 readiness score.
          </p>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-6 pb-20">
        <div className="grid gap-6 lg:grid-cols-3">
          {tiers.map((t) => (
            <div
              key={t.name}
              className={`relative flex flex-col rounded-2xl p-8 ${
                t.highlight
                  ? 'bg-ink-950 text-white shadow-2xl shadow-ink-900/20 ring-1 ring-ink-900'
                  : 'border border-ink-100 bg-white'
              }`}
            >
              {t.highlight && (
                <span className="absolute -top-3 left-8 rounded-full bg-brand-500 px-3 py-1 text-xs font-semibold text-white">Most popular</span>
              )}
              <h2 className={`text-lg font-semibold ${t.highlight ? 'text-white' : 'text-ink-950'}`}>{t.name}</h2>
              <p className={`mt-2 text-sm ${t.highlight ? 'text-ink-300' : 'text-ink-500'}`}>{t.blurb}</p>
              <p className="mt-6 flex items-baseline gap-1">
                <span className={`text-4xl font-bold tracking-tight ${t.highlight ? 'text-white' : 'text-ink-950'}`}>{t.price}</span>
                <span className={t.highlight ? 'text-ink-300' : 'text-ink-500'}>{t.period}</span>
              </p>
              <Link
                href={t.cta.href}
                className={`mt-6 rounded-xl px-4 py-3 text-center text-sm font-semibold transition ${
                  t.highlight
                    ? 'bg-brand-500 text-white hover:bg-brand-400'
                    : 'border border-ink-200 text-ink-900 hover:bg-ink-50'
                }`}
              >
                {t.cta.label}
              </Link>
              <ul className="mt-8 space-y-3 text-sm">
                {t.features.map((f) => (
                  <li key={f} className="flex gap-3">
                    <Check className={`h-5 w-5 flex-none ${t.highlight ? 'text-brand-400' : 'text-brand-600'}`} />
                    <span className={t.highlight ? 'text-ink-200' : 'text-ink-700'}>{f}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        <p className="mt-8 text-center text-sm text-ink-500">
          Prices exclude VAT. Want to see it with real data first?{' '}
          <Link href="/demo" className="font-semibold text-brand-700 hover:text-brand-800">Explore the live demo</Link>
        </p>
      </section>

      <section className="mx-auto max-w-3xl px-6 pb-12">
        <h2 className="text-center text-2xl font-bold tracking-tight text-ink-950">Pricing questions</h2>
        <dl className="mt-8 grid gap-6 sm:grid-cols-2">
          {faqs.map((f) => (
            <div key={f.q} className="rounded-2xl border border-ink-100 p-6">
              <dt className="font-semibold text-ink-950">{f.q}</dt>
              <dd className="mt-2 text-sm leading-relaxed text-ink-600">{f.a}</dd>
            </div>
          ))}
        </dl>
      </section>

      <FinalCta />
      <MarketingFooter />
    </div>
  );
}
