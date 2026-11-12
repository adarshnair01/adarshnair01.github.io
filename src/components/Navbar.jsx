import { Menu, X, Github, Linkedin } from 'lucide-react';
import { useState } from 'react';
import { FairyLights } from './Festive';

const Navbar = () => {
    const [isOpen, setIsOpen] = useState(false);

    const navLinks = [
        { name: 'About', href: '/#about' },
        { name: 'Journey', href: '/#experience' },
        { name: 'Projects', href: '/#projects' },
        { name: 'Games', href: '/#games' },
        { name: 'Blog', href: 'https://adarshnair.online/blog/' },
        { name: 'Travel', href: 'https://adarshnair.in/' },
        { name: 'Contact', href: '/#contact' },
    ];

    return (
        <nav className="fixed w-[92%] max-w-[1400px] z-50 top-0 left-1/2 -translate-x-1/2">
            <div className="bg-night/85 backdrop-blur-md rounded-b-2xl shadow-[0_8px_30px_rgba(0,0,0,0.5)] border-x border-b border-frost/15">
                <div className="w-full flex flex-wrap items-center justify-between mx-auto px-4 py-3">
                    <a href="#" className="flex items-center gap-2 group">
                        <span className="text-2xl" aria-hidden="true">🎄</span>
                        <span className="text-2xl font-display font-bold whitespace-nowrap text-snow tracking-wide group-hover:text-gold transition-colors">Adarsh Nair</span>
                    </a>
                    <div className="flex md:order-2 items-center gap-3">
                        <div className="hidden md:flex gap-4">
                            <a href="https://github.com/adarshnair01" target="_blank" rel="noopener noreferrer" aria-label="GitHub" className="text-text-muted hover:text-gold transition-colors">
                                <Github size={20} />
                            </a>
                            <a href="https://www.linkedin.com/in/adarshnair01" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn" className="text-text-muted hover:text-gold transition-colors">
                                <Linkedin size={20} />
                            </a>
                        </div>
                        <button
                            onClick={() => setIsOpen(!isOpen)}
                            type="button"
                            aria-expanded={isOpen}
                            className="inline-flex items-center p-2 w-10 h-10 justify-center text-text-muted rounded-lg md:hidden hover:bg-frost/10 focus:outline-none focus:ring-2 focus:ring-frost/40"
                        >
                            <span className="sr-only">Open main menu</span>
                            {isOpen ? <X /> : <Menu />}
                        </button>
                    </div>
                    <div className={`items-center justify-between w-full md:flex md:w-auto md:order-1 ${isOpen ? 'block' : 'hidden'}`}>
                        <ul className="flex flex-col p-4 md:p-0 mt-4 border border-frost/15 rounded-lg md:gap-7 md:flex-row md:mt-0 md:border-0">
                            {navLinks.map((link) => (
                                <li key={link.name}>
                                    <a
                                        href={link.href}
                                        className="relative block py-2 px-3 md:p-0 text-snow/80 rounded hover:bg-frost/10 md:hover:bg-transparent md:hover:text-gold transition-colors tracking-wide after:hidden md:after:block after:absolute after:left-0 after:-bottom-1 after:h-0.5 after:w-0 after:bg-holly hover:after:w-full after:transition-all"
                                        onClick={() => setIsOpen(false)}
                                    >
                                        {link.name}
                                    </a>
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>
            </div>
            {/* Fairy lights hanging under the navbar */}
            <FairyLights className="-mt-1 px-4" count={22} />
        </nav>
    );
};

export default Navbar;
