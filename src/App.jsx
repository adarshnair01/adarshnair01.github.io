import Navbar from './components/Navbar';
import Hero from './components/Hero';
import About from './components/About';
import Experience from './components/Experience';

import Projects from './components/Projects';
import QuickGames from './components/QuickGames';
import Footer from './components/Footer';
import MouseFollower from './components/MouseFollower';
import ItachiMascot from './components/ItachiMascot';

function App() {
  return (
    <div className="min-h-screen text-text-main selection:bg-primary selection:text-white overflow-x-hidden transition-colors duration-500">
      <MouseFollower />
      <Navbar />

      {/* Material Dark Background Layer */}
      <div className="fixed inset-0 -z-50 pointer-events-none">
        <div className="absolute inset-0 bg-[#121212]"></div>
        {/* Subtle Radial Gradient for Depth */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[600px] bg-primary/10 rounded-full blur-[120px] opacity-20"></div>
        {/* Magical Mist/Noise */}
        <div className="absolute inset-0 opacity-[0.03] bg-[url('https://grainy-gradients.vercel.app/noise.svg')]"></div>
      </div>

      {/* Main Container - Material Card styling optional, but keeping 90% layout */}
      <div className="w-[90%] max-w-[1600px] mx-auto shadow-2xl relative bg-[#121212]">
        <Hero />
        <About />
      </div>

      {/* Full Width Experience Section (Merged with About) */}
      <Experience />

      <div className="w-[90%] max-w-[1600px] mx-auto shadow-2xl relative bg-[#121212]">
        <Projects />
        <QuickGames />
        <Footer />
        <ItachiMascot />
      </div>
    </div>
  )
}

export default App;
