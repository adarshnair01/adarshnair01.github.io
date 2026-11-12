import Navbar from './components/Navbar';
import Hero from './components/Hero';
import About from './components/About';
import Experience from './components/Experience';
import Projects from './components/Projects';
import QuickGames from './components/QuickGames';
import Footer from './components/Footer';
import MouseFollower from './components/MouseFollower';
import GoldenDeerMascot from './components/GoldenDeerMascot';
import { Mandala } from './components/Ornaments';

function App() {
  return (
    <div className="min-h-screen text-text-main selection:bg-saffron selection:text-ink overflow-x-hidden">
      <MouseFollower />
      <Navbar />

      {/* Night-sky over Ayodhya: warm ink base, saffron dusk glow, slow mandala */}
      <div className="fixed inset-0 -z-50 pointer-events-none overflow-hidden">
        <div className="absolute inset-0 bg-ink" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(240,138,36,0.16),transparent_60%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom,rgba(15,59,74,0.35),transparent_65%)]" />
        <Mandala className="absolute -right-48 top-1/4 w-[700px] h-[700px] text-gold/[0.05] animate-spin-slower" />
        <Mandala className="absolute -left-56 bottom-0 w-[600px] h-[600px] text-saffron/[0.04] animate-spin-slower" />
      </div>

      <main>
        <div className="w-[92%] max-w-[1400px] mx-auto relative">
          <Hero />
          <About />
        </div>

        {/* Full-width Rama Setu journey */}
        <Experience />

        <div className="w-[92%] max-w-[1400px] mx-auto relative">
          <Projects />
          <QuickGames />
        </div>
      </main>

      <Footer />
      <GoldenDeerMascot />
    </div>
  );
}

export default App;
