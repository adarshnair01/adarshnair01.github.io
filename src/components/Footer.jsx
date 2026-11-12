import { Github, Linkedin, Mail } from 'lucide-react';
import { KandaHeading, Diya } from './Ornaments';

const Footer = () => {
    return (
        <footer id="contact" className="relative pt-20 pb-10 mt-10">
            <div className="temple-border opacity-50 absolute top-0 inset-x-0" />
            <div className="container mx-auto px-4 text-center">
                <KandaHeading
                    kanda="Uttara Kāṇḍa · The Message"
                    title="Let's Connect"
                    sanskrit="॥ उत्तरकाण्ड ॥"
                />
                <p className="text-text-muted mb-10 max-w-xl mx-auto text-lg font-serif italic">
                    As Hanuman carried Rama's message across the sea, your note will find its way to me.
                    Whether you have a question or just want to say hi, I'll try my best to get back to you!
                </p>

                <div className="flex justify-center gap-6 mb-14">
                    {[
                        { href: 'https://github.com/adarshnair01', label: 'GitHub', Icon: Github },
                        { href: 'https://www.linkedin.com/in/adarshnair01', label: 'LinkedIn', Icon: Linkedin },
                        { href: 'mailto:adarshnair01@gmail.com', label: 'Email', Icon: Mail },
                    ].map((item) => (
                        <a
                            key={item.label}
                            href={item.href}
                            target={item.href.startsWith('http') ? '_blank' : undefined}
                            rel="noopener noreferrer"
                            aria-label={item.label}
                            className="w-14 h-14 rounded-full border border-gold/40 flex items-center justify-center text-gold hover:text-ink hover:bg-saffron hover:border-saffron hover:scale-110 transition-all shadow-[0_0_20px_rgba(240,138,36,0.15)]"
                        >
                            <item.Icon size={24} />
                        </a>
                    ))}
                </div>

                <div className="flex justify-center gap-10 mb-6">
                    <Diya /><Diya /><Diya />
                </div>
                <p className="font-sanskrit text-saffron text-xl mb-2" lang="sa">॥ शुभमस्तु ॥</p>
                <p className="text-text-muted text-sm">Crafted with devotion by Adarsh Nair</p>
                <p className="text-text-muted/60 text-xs mt-2">© {new Date().getFullYear()} All rights reserved.</p>
            </div>
        </footer>
    );
};

export default Footer;
