import { motion } from 'framer-motion';
import { Camera, MapPin, BookOpen, Trophy, ArrowUpRight } from 'lucide-react';
import { FestiveHeading, Snowflake } from './Festive';

// What I'm good at, each backed by something I've shipped (details in Journey & Projects)
const strengths = [
    { name: 'Recommendation Engines', proof: 'Two-tower recommender serving 5M+ daily users' },
    { name: 'Propensity & Churn', proof: 'Propensity model lifted conversion by 10%' },
    { name: 'Customer Segmentation', proof: 'Segmentation raised marketing effectiveness by 10%' },
    { name: 'Time-Series Forecasting', proof: 'Forecasts for 2,000+ stores, 25% more accurate' },
    { name: 'Generative AI', proof: 'Building with LLMs and retrieval-augmented generation' },
    { name: 'Team Leadership', proof: 'Leading data science teams since 2021' },
];

const mlSkills = [
    'LLM', 'GenAI', 'Two-Tower Model', 'RAG',
    'Regression (Linear, Logistic)', 'Kalman Filters', 'Decision Tree',
    'Support Vector Machine', 'K-Nearest Neighbors', 'Random Forest',
    'Gradient Boosting', 'AdaBoost', 'PCA', 'Neural Networks', 'Optimization', 'OLS', 'Deep Learning'
];

const interests = [
    {
        title: 'Photography',
        url: 'https://images.unsplash.com/photo-1516035069371-29a1b244cc32?auto=format&fit=crop&w=400&q=70',
        href: 'https://www.instagram.com/adarshnair01/',
        icon: Camera,
    },
    {
        title: 'Travel',
        url: 'https://images.unsplash.com/photo-1476514525535-07fb3b4ae5f1?auto=format&fit=crop&w=400&q=70',
        href: 'https://adarshnair.in/',
        icon: MapPin,
    },
    {
        title: 'Soccer',
        url: 'https://images.unsplash.com/photo-1579952363873-27f3bade9f55?auto=format&fit=crop&w=400&q=70',
        icon: Trophy,
    },
    {
        title: 'Reading',
        url: 'https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?auto=format&fit=crop&w=400&q=70',
        href: 'https://adarshnair.online/blog/',
        icon: BookOpen,
    },
];

const InterestTile = ({ item, index }) => {
    const Icon = item.icon;
    const inner = (
        <>
            <img
                src={item.url}
                alt=""
                loading="lazy"
                decoding="async"
                onError={(e) => { e.currentTarget.style.display = 'none'; }}
                className="absolute inset-0 w-full h-full object-cover transition-transform duration-700 group-hover:scale-110 opacity-40 group-hover:opacity-60"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-night via-night/50 to-transparent"></div>
            {/* Gift ribbon */}
            <span className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-3 bg-holly/70 group-hover:bg-holly transition-colors" aria-hidden="true" />
            <span className="absolute inset-x-0 top-1/3 h-3 bg-holly/70 group-hover:bg-holly transition-colors" aria-hidden="true" />

            <div className="absolute inset-0 p-3 flex flex-col justify-end">
                <div className="flex items-center gap-2 bg-night/70 rounded-md px-2 py-1 w-fit">
                    <Icon size={14} className="text-gold drop-shadow-md" aria-hidden="true" />
                    <span className="text-snow text-xs lg:text-sm font-bold tracking-wide leading-tight">{item.title}</span>
                    {item.href && <ArrowUpRight size={12} className="text-snow/70" aria-hidden="true" />}
                </div>
            </div>
        </>
    );
    const className = 'group relative block bg-night-card rounded-xl overflow-hidden shadow-md border border-frost/15 transition-all h-[120px]';

    return item.href ? (
        <motion.a
            href={item.href}
            target="_blank"
            rel="noopener noreferrer"
            whileHover={{ scale: 1.03, rotate: index % 2 ? 1 : -1 }}
            className={`${className} hover:border-gold/70`}
        >
            {inner}
        </motion.a>
    ) : (
        <div className={className}>{inner}</div>
    );
};

const About = () => {
    return (
        <section id="about" className="py-24 bg-transparent relative">
            <div className="container mx-auto px-4">

                <FestiveHeading
                    eyebrow="Chapter One"
                    title="About Me"
                    script="the story so far"
                />

                <div className="grid grid-cols-1 lg:grid-cols-4 gap-8 items-start">

                    {/* Bio & Interests */}
                    <motion.div
                        initial={{ opacity: 0, x: -50 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.2 }}
                        className="lg:col-span-2"
                    >
                        <h3 className="text-2xl font-bold text-snow mb-6 flex items-center gap-3">
                            <Snowflake className="w-6 h-6 text-frost" /> Who I Am
                        </h3>
                        {/* Gift tag */}
                        <div className="relative bg-cream text-cream-ink p-6 md:p-8 rounded-xl shadow-[0_10px_40px_rgba(0,0,0,0.5)] border-l-[10px] border-holly">
                            <span className="absolute -left-[22px] top-6 w-6 h-6 rounded-full bg-night border-4 border-gold" aria-hidden="true" />
                            <p className="font-script text-2xl text-holly mb-2">To: you, From: Adarsh</p>
                            <p className="text-lg leading-relaxed">
                                I am a data science professional from Bengaluru, India, with a passion for extracting actionable insights from complex datasets.
                                My expertise spans across machine learning, predictive modeling, and software architecture.
                                I thrive on solving challenging problems, from improving sales forecasting to building large-scale recommendation systems.
                            </p>
                        </div>

                        {/* Personal Interests */}
                        <div className="mt-10">
                            <h3 className="text-lg font-bold text-snow/90 mb-4 flex items-center gap-2">
                                <span aria-hidden="true">⭐</span> Personal Interests
                            </h3>
                            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                                {interests.map((item, index) => (
                                    <InterestTile key={item.title} item={item} index={index} />
                                ))}
                            </div>
                        </div>
                    </motion.div>

                    {/* Strengths with proof */}
                    <motion.div
                        initial={{ opacity: 0, y: 30 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.3 }}
                        className="lg:col-span-1"
                    >
                        <h3 className="text-2xl font-bold text-snow mb-1">What I Do</h3>
                        <p className="font-script text-2xl text-gold mb-4">gifts I bring</p>
                        <ul className="space-y-4 frost-card snow-cap p-6">
                            {strengths.map((s) => (
                                <li key={s.name} className="group">
                                    <p className="text-sm font-bold text-snow group-hover:text-gold transition-colors">{s.name}</p>
                                    <p className="text-xs text-text-muted leading-snug mt-0.5">{s.proof}</p>
                                    <div className="mt-2 h-1 rounded-full candy-stripe opacity-70 group-hover:opacity-100 transition-opacity" aria-hidden="true" />
                                </li>
                            ))}
                        </ul>
                    </motion.div>

                    {/* Machine Learning */}
                    <motion.div
                        initial={{ opacity: 0, x: 50 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.4 }}
                        className="lg:col-span-1"
                    >
                        <h3 className="text-2xl font-bold text-snow mb-1">Machine Learning</h3>
                        <p className="font-script text-2xl text-gold mb-4">my toolkit</p>
                        <ul className="flex flex-wrap gap-2 frost-card snow-cap p-6">
                            {mlSkills.map((skill, index) => (
                                <motion.li
                                    key={skill}
                                    initial={{ opacity: 0, scale: 0.8 }}
                                    whileInView={{ opacity: 1, scale: 1 }}
                                    viewport={{ once: true }}
                                    transition={{ delay: index * 0.05 + 0.5 }}
                                    className={`px-3 py-1 rounded-full text-[11px] font-bold border transition-colors cursor-default ${index % 3 === 0 ? 'bg-holly/15 text-[#ff8a99] border-holly/30 hover:bg-holly/30' : index % 3 === 1 ? 'bg-pine/15 text-[#6fe0a6] border-pine/30 hover:bg-pine/30' : 'bg-gold/10 text-gold border-gold/30 hover:bg-gold/25'}`}
                                >
                                    {skill}
                                </motion.li>
                            ))}
                        </ul>
                    </motion.div>

                </div>
            </div>
        </section>
    );
};

export default About;
