import { Menu, X, Github, Linkedin } from 'lucide-react';
import { useState } from 'react';

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
        <nav className="fixed w-[92%] max-w-[1400px] z-50 top-0 left-1/2 -translate-x-1/2 bg-ink/85 backdrop-blur-md rounded-b-xl shadow-[0_8px_30px_rgba(0,0,0,0.5)] border-x border-b border-gold/20">
            <div className="w-full flex flex-wrap items-center justify-between mx-auto px-4 py-3">
                <a href="#" className="flex items-center gap-3 group">
                    <span className="font-sanskrit text-2xl text-saffron group-hover:text-gold transition-colors" lang="sa" aria-hidden="true">ॐ</span>
                    <span className="text-2xl whitespace-nowrap text-palm font-epic tracking-wide">Adarsh Nair</span>
                </a>
                <div className="flex md:order-2 items-center gap-3">
                    <div className="hidden md:flex gap-4">
                        <a href="https://github.com/adarshnair01" target="_blank" rel="noopener noreferrer" aria-label="GitHub" className="text-text-muted hover:text-saffron transition-colors">
                            <Github size={20} />
                        </a>
                        <a href="https://www.linkedin.com/in/adarshnair01" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn" className="text-text-muted hover:text-saffron transition-colors">
                            <Linkedin size={20} />
                        </a>
                    </div>
                    <button
                        onClick={() => setIsOpen(!isOpen)}
                        type="button"
                        aria-expanded={isOpen}
                        className="inline-flex items-center p-2 w-10 h-10 justify-center text-text-muted rounded-lg md:hidden hover:bg-gold/10 focus:outline-none focus:ring-2 focus:ring-gold/40"
                    >
                        <span className="sr-only">Open main menu</span>
                        {isOpen ? <X /> : <Menu />}
                    </button>
                </div>
                <div className={`items-center justify-between w-full md:flex md:w-auto md:order-1 ${isOpen ? 'block' : 'hidden'}`}>
                    <ul className="flex flex-col p-4 md:p-0 mt-4 border border-gold/20 rounded-lg md:gap-7 md:flex-row md:mt-0 md:border-0">
                        {navLinks.map((link) => (
                            <li key={link.name}>
                                <a
                                    href={link.href}
                                    className="relative block py-2 px-3 md:p-0 text-palm/80 rounded hover:bg-gold/10 md:hover:bg-transparent md:hover:text-saffron transition-colors tracking-wide after:hidden md:after:block after:absolute after:left-0 after:-bottom-1 after:h-px after:w-0 after:bg-saffron hover:after:w-full after:transition-all"
                                    onClick={() => setIsOpen(false)}
                                >
                                    {link.name}
                                </a>
                            </li>
                        ))}
                    </ul>
                </div>
            </div>
            <div className="temple-border opacity-40" />
        </nav>
    );
};

export default Navbar;
