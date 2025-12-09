import { Github, Linkedin, Mail, Heart } from 'lucide-react';

const Footer = () => {
    return (
        <footer id="contact" className="bg-transparent py-12">
            <div className="container mx-auto px-4 text-center">
                <h2 className="text-3xl font-bold text-white mb-8 font-harry tracking-wider">Let's Connect</h2>
                <p className="text-gray-400 mb-8 max-w-xl mx-auto font-medium">
                    I'm currently looking for new opportunities. Whether you have a question or just want to say hi, I'll try my best to get back to you!
                </p>

                <div className="flex justify-center gap-8 mb-12">
                    <a href="https://github.com/adarshnair01" className="text-gray-500 hover:text-primary hover:scale-110 transition-all">
                        <Github size={32} />
                    </a>
                    <a href="https://www.linkedin.com/in/adarshnair01" className="text-gray-500 hover:text-[#0077b5] hover:scale-110 transition-all">
                        <Linkedin size={32} />
                    </a>
                    <a href="mailto:adarshnair01@gmail.com" className="text-gray-500 hover:text-red-600 hover:scale-110 transition-all">
                        <Mail size={32} />
                    </a>
                </div>

                <div className="text-gray-500 text-sm flex items-center justify-center gap-1 font-medium">
                    <p>Created with</p>
                    <Heart size={14} className="text-red-500 fill-red-500 animate-pulse" />
                    <p>by Adarsh Nair</p>
                </div>
                <p className="text-gray-600 text-xs mt-2">© {new Date().getFullYear()} All rights reserved.</p>
            </div>
        </footer>
    );
};

export default Footer;
