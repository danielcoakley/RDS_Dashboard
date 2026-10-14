const faqs = [
  ['Does using EnMS guarantee ISO 50001 certification?', 'No software can guarantee certification — that’s decided by your accredited certification body. EnMS gives you the system, data and evidence the standard requires, organised clause by clause.'],
  ['How is the baseline weather-normalised?', 'We fetch daily temperatures for each site, calculate heating and cooling degree days (HDD/CDD), and fit a regression model per meter. That lets you compare performance fairly across mild and harsh years, as clause 6.5 expects.'],
  ['What data do I need to get started?', 'Meter consumption data as a CSV — monthly or daily — with a date, meter name and consumption. Site addresses are used to fetch weather automatically.'],
  ['How are SEUs identified?', 'Meters are ranked by share of consumption and improvement potential, and those above your threshold are flagged as significant energy uses for review.'],
  ['Can we use it across multiple sites?', 'Yes. Professional covers up to 10 sites, and Enterprise supports unlimited sites with group-level reporting.'],
  ['Is there a free trial?', 'Yes — 14 days with full features and no credit card required.'],
];

export default function FAQ() {
  return (
    <div className="divide-y divide-white/10 border-y border-white/10">
      {faqs.map(([q, a]) => (
        <details key={q} className="group py-5">
          <summary className="flex cursor-pointer list-none items-center justify-between gap-4 font-medium text-white">
            {q}
            <span className="font-mono text-brand-400 transition group-open:rotate-45">+</span>
          </summary>
          <p className="mt-3 leading-relaxed text-ink-300">{a}</p>
        </details>
      ))}
    </div>
  );
}
