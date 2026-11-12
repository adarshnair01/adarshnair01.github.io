import { Github, Linkedin, Mail, Instagram } from 'lucide-react';
import { LINKS } from '../siteConfig';
import { FestiveHeading, FairyLights } from './Festive';

const Footer = () => {
    const nextYear = new Date().getMonth() === 0 ? new Date().getFullYear() : new Date().getFullYear() + 1;

    return (
        <footer id="contact" className="relative pt-16 pb-10 mt-10">
            <FairyLights className="absolute top-0 inset-x-0" count={26} />
            <div className="container mx-auto px-4 text-center pt-8">
                <FestiveHeading
                    eyebrow="The Last Chapter"
                    title="Let's Connect"
                    script="send a greeting"
                />
                <p className="text-text-muted mb-8 max-w-xl mx-auto text-lg">
                    Have a data science problem, a role, or an idea you'd like to talk through?
                    My inbox is open and I'll get back to you as soon as I can.
                </p>

                <a
                    href={LINKS.email}
                    className="inline-flex items-center gap-2 px-8 py-3 mb-3 bg-holly hover:bg-holly-deep text-snow rounded-full font-bold transition-all hover:shadow-[0_0_25px_rgba(212,36,59,0.55)] uppercase tracking-[0.15em] text-sm"
                >
                    <Mail size={18} aria-hidden="true" /> Email Me
                </a>
                <p className="text-text-muted text-sm mb-10">{LINKS.email.replace('mailto:', '')}</p>

                <div className="flex justify-center gap-6 mb-14">
                    {[
                        { href: LINKS.github, label: 'GitHub', Icon: Github },
                        { href: LINKS.linkedin, label: 'LinkedIn', Icon: Linkedin },
                        { href: LINKS.instagram, label: 'Instagram', Icon: Instagram },
                    ].map((item) => (
                        <a
                            key={item.label}
                            href={item.href}
                            target={item.href.startsWith('http') ? '_blank' : undefined}
                            rel="noopener noreferrer"
                            aria-label={item.label}
                            className="w-14 h-14 rounded-full border-2 border-frost/30 flex items-center justify-center text-snow hover:bg-holly hover:border-holly hover:scale-110 transition-all shadow-[0_0_20px_rgba(169,212,245,0.12)]"
                        >
                            <item.Icon size={24} />
                        </a>
                    ))}
                </div>

                <p className="font-script text-gold text-4xl mb-2">Merry Christmas &amp; a Happy {nextYear}</p>
                <p className="text-text-muted text-sm">Made with warm cocoa ☕ by Adarsh Nair</p>
                <p className="text-text-muted/60 text-xs mt-2">© {new Date().getFullYear()} All rights reserved.</p>
            </div>
        </footer>
    );
};

export default Footer;
