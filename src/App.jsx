import Navbar from './components/Navbar';
import Hero from './components/Hero';
import About from './components/About';
import Experience from './components/Experience';
import Projects from './components/Projects';
import QuickGames from './components/QuickGames';
import Footer from './components/Footer';
import MouseFollower from './components/MouseFollower';
import SnowmanMascot from './components/SnowmanMascot';
import { MotionConfig } from 'framer-motion';
import { Snowfall } from './components/Festive';

function App() {
  return (
    <MotionConfig reducedMotion="user">
    <div id="top" className="min-h-screen text-text-main selection:bg-holly selection:text-snow overflow-x-hidden">
      <MouseFollower />
      <Snowfall />
      <Navbar />

      {/* Winter night sky: deep navy, aurora glow, warm light near the horizon */}
      <div className="fixed inset-0 -z-50 pointer-events-none overflow-hidden">
        <div className="absolute inset-0 bg-night" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,rgba(31,138,85,0.18),transparent_55%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_left,rgba(169,212,245,0.10),transparent_50%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom,rgba(212,36,59,0.10),transparent_60%)]" />
      </div>

      <main>
        <div className="w-[92%] max-w-[1400px] mx-auto relative">
          <Hero />
          <About />
        </div>

        {/* Full-width journey band */}
        <Experience />

        <div className="w-[92%] max-w-[1400px] mx-auto relative">
          <Projects />
          <QuickGames />
        </div>
      </main>

      <Footer />
      <SnowmanMascot />
    </div>
    </MotionConfig>
  );
}

export default App;
