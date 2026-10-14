import Nav from '@/components/marketing/Nav';
import Footer from '@/components/marketing/Footer';
import { MarketingShell } from '@/components/marketing/ui';
import { Hero, Problem, HowItWorks, FeatureGrid, Readiness, Quotes, FAQSection, FinalCTA } from '@/components/marketing/LandingSections';

export default function LandingPage() {
  return (
    <MarketingShell>
      <Nav />
      <main>
        <Hero />
        <Problem />
        <HowItWorks />
        <FeatureGrid />
        <Readiness />
        <Quotes />
        <FAQSection />
        <FinalCTA />
      </main>
      <Footer />
    </MarketingShell>
  );
}
