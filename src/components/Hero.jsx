import { motion } from 'framer-motion';

const Hero = () => {
    return (
        <section className="min-h-screen flex items-center justify-center pt-16 relative overflow-hidden">
            {/* Background Gradients */}
            <div className="absolute top-0 left-0 w-full h-full overflow-hidden -z-10">
                <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-primary/20 rounded-full blur-[100px] animate-pulse"></div>
                <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-secondary/20 rounded-full blur-[100px] animate-pulse delay-1000"></div>
            </div>

            <div className="container mx-auto px-4 text-center z-10">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8 }}
                >
                    <h2 className="text-xl md:text-2xl text-primary font-semibold mb-4 tracking-wide font-harry">HELLO, I AM</h2>
                    <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 tracking-wider font-harry drop-shadow-lg">
                        Adarsh Nair
                    </h1>
                    <h3 className="text-2xl md:text-4xl text-gray-400 mb-8 font-light">
                        Data Science Manager & <br className="md:hidden" />
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-amber-500 font-semibold">
                            ML Engineer
                        </span>
                    </h3>

                    <p className="max-w-2xl mx-auto text-gray-400 text-lg mb-10 leading-relaxed font-body">
                        I lead teams to build predictive models and recommendation engines.
                        Passionate about turning data into actionable insights and scalable solutions.
                    </p>

                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.5, duration: 0.5 }}
                        className="flex justify-center gap-6"
                    >
                        <a
                            href="#projects"
                            className="px-8 py-3 bg-primary hover:bg-primary/90 text-white rounded-full font-medium transition-all hover:shadow-[0_0_20px_rgba(211,47,47,0.4)] shadow-lg uppercase tracking-wider text-sm flex items-center justify-center"
                        >
                            View Works
                        </a>
                        <a
                            href="#contact"
                            className="px-8 py-3 border-2 border-primary/50 text-white hover:border-primary hover:text-primary rounded-full font-medium transition-all hover:bg-white/5 uppercase tracking-wider text-sm flex items-center justify-center"
                        >
                            Contact Me
                        </a>
                    </motion.div>
                </motion.div>
            </div>

            {/* Scroll indicator */}
            <motion.div
                className="absolute bottom-8 left-1/2 -translate-x-1/2 text-gray-500"
                animate={{ y: [0, 10, 0] }}
                transition={{ duration: 1.5, repeat: Infinity }}
            >
                <div className="w-6 h-10 border-2 border-gray-500 rounded-full flex justify-center p-2">
                    <div className="w-1 h-2 bg-gray-500 rounded-full"></div>
                </div>
            </motion.div>
        </section>
    );
};

export default Hero;
