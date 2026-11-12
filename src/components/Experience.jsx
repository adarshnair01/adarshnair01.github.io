import { motion } from 'framer-motion';
import { GraduationCap } from 'lucide-react';
import { Bauble } from './Festive';

const BAUBLE_COLORS = ['#d4243b', '#f0c34e', '#1f8a55', '#a9d4f5', '#d4243b', '#f0c34e'];

// Wrapped present (used under the tree in the teaser)
const Present = ({ box, ribbon, className = '' }) => (
    <div className={`relative rounded-sm shadow-[0_8px_16px_rgba(0,0,0,0.4)] ${className}`} style={{ backgroundColor: box }} aria-hidden="true">
        <span className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-2" style={{ backgroundColor: ribbon }} />
        <span className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-2" style={{ backgroundColor: ribbon }} />
        <span className="absolute -top-2 left-1/2 -translate-x-1/2 w-5 h-3 rounded-full border-2" style={{ borderColor: ribbon }} />
    </div>
);

// Simple decorated Christmas tree
const Tree = ({ className = '' }) => (
    <svg viewBox="0 0 120 160" className={className} aria-hidden="true">
        <path d="M60 4 L104 70 H84 L112 112 H88 L118 150 H2 L32 112 H8 L36 70 H16 Z" fill="#1f8a55" stroke="#0f5132" strokeWidth="2" />
        <rect x="52" y="150" width="16" height="10" fill="#5a3522" />
        <path d="M60 0 L64 10 L74 10 L66 16 L69 26 L60 20 L51 26 L54 16 L46 10 L56 10 Z" fill="#f0c34e" />
        {[[40, 60, '#d4243b'], [78, 66, '#f0c34e'], [56, 92, '#a9d4f5'], [30, 104, '#f0c34e'], [88, 108, '#d4243b'], [46, 136, '#d4243b'], [74, 138, '#a9d4f5'], [60, 48, '#d4243b']].map(([x, y, c], i) => (
            <circle key={i} cx={x} cy={y} r="5" fill={c} className="animate-twinkle" style={{ animationDelay: `${i * 0.3}s` }} />
        ))}
    </svg>
);

const experience = [
    {
        role: 'Senior Data Science Manager',
        company: 'Yum! Brands',
        period: '2022 - Present',
        desc: 'Manage a team of data scientists and analysts to build customer propensity and churn models, as well as a recommendation engine to enhance customer experience and revenue growth for a leading restaurant brand.',
    },
    {
        role: 'Data Science Manager',
        company: 'Yum! Brands',
        period: '2021 - 2022',
        desc: 'Led a team of data scientists and analysts to develop predictive models, segmentation strategies, and pricing optimization algorithms, resulting in a 10% increase in conversion rates through a propensity model and a 10% increase in marketing effectiveness through customer segmentation.',
    },
    {
        role: 'Principal Data Scientist',
        company: 'Kvantum Inc.',
        period: '2020 - 2021',
        desc: 'Managed end-to-end development of a sales forecasting system with cross-functional teams, improving forecast accuracy by 25%, while conducting data analysis and creating visualizations for technical and non-technical stakeholders.',
    },
    {
        role: 'Senior Data Scientist',
        company: 'Kvantum Inc.',
        period: '2019 - 2020',
        desc: 'Lead the end-to-end design, development, and deployment of time series predictive modeling solutions.',
    },
    {
        role: 'Data Scientist',
        company: 'Kvantum Inc.',
        period: '2017 - 2019',
        desc: 'Monitored technical aspects of the delivery for several projects and develop new NLP components.',
    },
    {
        role: 'Software Developer',
        company: 'MAQ Software',
        period: '2016 - 2017',
        desc: 'Design software architecture, Develop algorithms and Ensure quality of code.',
    }
];

const education = [
    {
        degree: "Master's Degree in Information Technology",
        school: "Indian Institute of Information Technology, Allahabad (2014 - 2016)",
    },
    {
        degree: "Bachelor's Degree in Information Technology",
        school: "Government Engineering College, Bhavnagar (2009 - 2013)",
    }
];

const Experience = () => {
    return (
        <section id="experience" className="relative overflow-hidden py-10">

            {/* Decorative header band: presents under the tree, one per role */}
            <div className="relative w-full h-[320px] overflow-hidden bg-gradient-to-b from-night-soft via-[#13284a] to-night-soft border-y-4 border-holly flex flex-col items-center justify-center shadow-[0_0_60px_rgba(0,0,0,0.6)]">
                <div className="absolute inset-x-0 top-0 h-3 candy-stripe opacity-80" aria-hidden="true" />
                <div className="absolute inset-x-0 bottom-0 h-16 bg-snow rounded-t-[100%]" aria-hidden="true" />

                <Tree className="absolute bottom-6 left-[6%] md:left-[12%] w-28 md:w-40 h-auto" />
                <Tree className="absolute bottom-6 right-[6%] md:right-[12%] w-24 md:w-32 h-auto hidden sm:block" />

                <div className="absolute bottom-8 inset-x-0 flex justify-center items-end gap-3 md:gap-5" aria-hidden="true">
                    {experience.map((_, i) => (
                        <motion.div
                            key={i}
                            animate={{ y: [0, -5, 0] }}
                            transition={{ duration: 2.4 + (i % 3) * 0.5, repeat: Infinity, delay: i * 0.25 }}
                            className={i > 3 ? 'hidden md:block' : ''}
                        >
                            <Present
                                box={['#d4243b', '#1f8a55', '#f0c34e', '#a9d4f5', '#d4243b', '#1f8a55'][i]}
                                ribbon={['#f0c34e', '#d4243b', '#1f8a55', '#d4243b', '#f4f7fc', '#f0c34e'][i]}
                                className={['w-12 h-10', 'w-14 h-12', 'w-10 h-9', 'w-14 h-10', 'w-12 h-12', 'w-10 h-10'][i]}
                            />
                        </motion.div>
                    ))}
                </div>

                <p className="section-eyebrow relative z-10 mb-2">Chapter Two</p>
                <h2 className="relative z-10 text-4xl md:text-6xl font-display font-bold text-snow mb-1 tracking-wide text-center">
                    My Journey
                </h2>
                <p className="relative z-10 font-script text-gold text-3xl md:text-4xl mb-24 px-4 text-center">
                    every year, a new ornament
                </p>
            </div>

            <div className="relative container mx-auto px-4 pt-20 pb-6 max-w-5xl">
                {/* A string of lights runs down the middle; each role hangs from a bauble */}
                <div className="relative">
                    <div className="absolute left-7 md:left-1/2 top-0 bottom-0 w-0.5 -translate-x-1/2 bg-[#2c3a2f]" aria-hidden="true" />

                    <ol className="space-y-12 relative">
                        {experience.map((item, index) => {
                            const left = index % 2 === 0;
                            return (
                                <motion.li
                                    key={index}
                                    initial={{ opacity: 0, y: 60 }}
                                    whileInView={{ opacity: 1, y: 0 }}
                                    viewport={{ once: true, margin: '-60px' }}
                                    transition={{ duration: 0.6, type: 'spring', bounce: 0.35 }}
                                    className={`relative pl-20 md:pl-0 md:w-1/2 ${left ? 'md:pr-16 md:text-right' : 'md:ml-auto md:pl-16'}`}
                                >
                                    <Bauble
                                        color={BAUBLE_COLORS[index % BAUBLE_COLORS.length]}
                                        className={`absolute z-10 top-0 w-12 h-16 drop-shadow-[0_0_12px_rgba(240,195,78,0.35)] ${left ? 'left-1 md:left-auto md:-right-6' : 'left-1 md:-left-6'}`}
                                    />

                                    <div className="frost-card snow-cap p-6 md:p-7 !bg-night-card">
                                        <p className="text-gold text-xs font-bold uppercase tracking-[0.25em] mb-1">{item.period}</p>
                                        <h3 className="text-2xl font-bold text-snow leading-tight mb-1">{item.role}</h3>
                                        <p className="text-[#ff8a99] font-bold text-sm uppercase tracking-widest mb-3">{item.company}</p>
                                        <p className="text-base leading-relaxed text-snow/80">{item.desc}</p>
                                    </div>
                                </motion.li>
                            );
                        })}
                    </ol>

                    {/* Education */}
                    <div className="relative mt-20 pt-6">
                        <h3 className="text-3xl font-bold text-snow mb-1 flex items-center justify-center gap-3 text-center">
                            <GraduationCap className="text-gold" size={30} aria-hidden="true" /> Education
                        </h3>
                        <p className="text-center font-script text-gold text-2xl mb-8">where the first gifts were made</p>
                        <div className="grid md:grid-cols-2 gap-6">
                            {education.map((item, index) => (
                                <motion.div
                                    key={index}
                                    initial={{ opacity: 0, y: 20 }}
                                    whileInView={{ opacity: 1, y: 0 }}
                                    viewport={{ once: true }}
                                    transition={{ delay: index * 0.15 }}
                                    className="bg-cream text-cream-ink p-6 rounded-xl border-t-8 border-pine shadow-lg"
                                >
                                    <h4 className="text-xl font-bold mb-2">{item.degree}</h4>
                                    <p className="text-holly-deep text-sm font-bold uppercase tracking-wider">{item.school}</p>
                                </motion.div>
                            ))}
                        </div>
                    </div>
                </div>

                <p className="mt-16 text-center font-script text-gold text-3xl">and the best is yet to come…</p>
            </div>
        </section>
    );
};

export default Experience;
