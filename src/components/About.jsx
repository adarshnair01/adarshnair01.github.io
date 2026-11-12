import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import { Camera, MapPin, Smile } from 'lucide-react';
import FullScreenOverlay from './FullScreenOverlay';
import { KandaHeading, Lotus } from './Ornaments';

// Skill bar glows like a sacrificial fire (agni) on hover
const agniVariant = {
    rest: { filter: "brightness(1)" },
    hover: {
        filter: "brightness(1.35)",
        transition: { duration: 0.3 }
    }
};
const INSTAGRAM_TOKEN = "";
// ----------------------------------------------------------------------

const About = () => {
    const [isInstagramOpen, setIsInstagramOpen] = useState(false);
    const [instagramFeed, setInstagramFeed] = useState([]);

    const skills = [
        { name: 'Recommendation Engine', level: '95%' },
        { name: 'Generative AI', level: '81%' },
        { name: 'Segmentation', level: '85%' },
        { name: 'Propensity and Churn', level: '90%' },
        { name: 'Predictive Modelling', level: '92%' },
        { name: 'Time Series Modelling', level: '88%' },
        { name: 'Data Visualization', level: '85%' },
        { name: 'Hyperparameter Tuning', level: '82%' },
    ];

    // Interests data (Fallback Mock Data)
    const interests = [
        {
            title: 'Photography',
            url: 'https://images.unsplash.com/photo-1516035069371-29a1b244cc32?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
            desc: 'Capturing moments and landscapes with a focus on street photography.',
            icon: Camera
        },
        {
            title: 'Travel',
            url: 'https://images.unsplash.com/photo-1476514525535-07fb3b4ae5f1?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
            desc: 'Exploring new cultures and cuisines. Visited 10+ states in India.',
            icon: MapPin
        },
        {
            title: 'Soccer',
            url: 'https://images.unsplash.com/photo-1579952363873-27f3bade9f55?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
            desc: 'Passionate about football. Avid supporter and weekend player.',
            icon: Smile
        },
        {
            title: 'Reading',
            url: 'https://images.unsplash.com/photo-1524995997946-a1c2e315a42f?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
            desc: (
                <>
                    Enjoying sci-fi novels and{' '}
                    <a href="https://adarshnair.online/blog/" target="_blank" rel="noopener noreferrer" className="text-secondary hover:underline">technical blogs</a>.
                </>
            ),
            icon: Smile
        }
    ];

    const mlSkills = [
        'LLM', 'GenAI', 'Two-Tower Model', 'RAG',
        'Regression (Linear, Logistic)', 'Kalman Filters', 'Decision Tree',
        'Support Vector Machine', 'K-Nearest Neighbors', 'Random Forest',
        'Gradient Boosting', 'AdaBoost', 'PCA', 'Neural Networks', 'Optimization', 'OLS', 'Deep Learning'
    ];

    // Fetch Instagram Data (Effect)
    useEffect(() => {
        if (!INSTAGRAM_TOKEN) {
            setInstagramFeed(interests); // Use mock data if no token
            return;
        }

        const fetchInstagram = async () => {
            // API Endpoint for User Media
            const url = `https://graph.instagram.com/me/media?fields=id,caption,media_type,media_url,thumbnail_url,permalink&access_token=${INSTAGRAM_TOKEN}`;

            try {
                const response = await fetch(url);
                const data = await response.json();

                if (data && data.data) {
                    // Map API response to our format
                    const validPosts = data.data.slice(0, 9).map(post => ({
                        id: post.id,
                        url: post.media_type === 'VIDEO' ? post.thumbnail_url : post.media_url,
                        permalink: post.permalink,
                        caption: post.caption || 'Instagram Post'
                    }));
                    setInstagramFeed(validPosts);
                } else {
                    console.warn("Instagram API Error:", data);
                    setInstagramFeed(interests); // Fallback on error
                }
            } catch (error) {
                console.error("Failed to fetch Instagram:", error);
                setInstagramFeed(interests); // Fallback on network error
            }
        };

        fetchInstagram();
    }, []);

    // Helper to determine what to render in the Grid
    // If using API, we show up to 9 posts. If mock, we show the 4 interests + placeholders.
    const displayPosts = INSTAGRAM_TOKEN ? instagramFeed : interests.slice(0, 3);
    const showPlaceholders = !INSTAGRAM_TOKEN; // Only show 'Post' placeholders if using mock data

    return (
        <section id="about" className="py-24 bg-transparent relative">
            <div className="container mx-auto px-4">

                <KandaHeading
                    kanda="Bāla Kāṇḍa · The Beginning"
                    title="About Me"
                    sanskrit="॥ बालकाण्ड ॥"
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
                        <h3 className="text-2xl text-gold mb-6 flex items-center gap-3">
                            <Lotus className="w-8 h-5 text-saffron" /> Who I Am
                        </h3>
                        <div className="palm-leaf text-palm-ink p-6 md:p-8 rounded-sm shadow-[0_10px_40px_rgba(0,0,0,0.5)] relative">
                            {/* Binding holes of a palm-leaf manuscript */}
                            <span className="absolute left-4 top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full bg-ink/70" aria-hidden="true" />
                            <span className="absolute right-4 top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full bg-ink/70" aria-hidden="true" />
                            <p className="font-serif text-lg md:text-xl leading-relaxed px-4">
                                I am a data science professional from Bengaluru, India, with a passion for extracting actionable insights from complex datasets.
                                My expertise spans across machine learning, predictive modeling, and software architecture.
                                I thrive on solving challenging problems, from improving sales forecasting to building large-scale recommendation systems.
                            </p>
                        </div>

                        {/* Personal Interests */}
                        <div className="mt-10">
                            <h4 className="text-lg text-palm/90 mb-4 flex items-center gap-2 font-epic">
                                <span className="text-saffron" aria-hidden="true">✺</span> Personal Interests
                            </h4>
                            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                                {interests.map((item, index) => (
                                    <motion.button
                                        type="button"
                                        key={index}
                                        whileHover={{ scale: 1.03 }}
                                        onClick={() => setIsInstagramOpen(true)}
                                        className="group relative bg-ink-card rounded-t-[40px] rounded-b-md overflow-hidden shadow-md border border-gold/25 hover:border-saffron/70 transition-all h-[120px] text-left"
                                    >
                                        <div
                                            className="absolute inset-0 bg-cover bg-center transition-transform duration-700 group-hover:scale-110 opacity-40 group-hover:opacity-60 sepia-[.4]"
                                            style={{ backgroundImage: `url(${item.url})` }}
                                        ></div>
                                        <div className="absolute inset-0 bg-gradient-to-t from-ink via-ink/50 to-transparent"></div>

                                        <div className="absolute inset-0 p-3 flex flex-col justify-end">
                                            <div className="flex items-center gap-2">
                                                <item.icon size={14} className="text-saffron drop-shadow-md" />
                                                <h5 className="text-palm text-xs lg:text-sm tracking-wide font-epic drop-shadow-md leading-tight">{item.title}</h5>
                                            </div>
                                        </div>
                                    </motion.button>
                                ))}
                            </div>
                        </div>
                    </motion.div>

                    {/* Skills — the Astras */}
                    <motion.div
                        initial={{ opacity: 0, y: 30 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.3 }}
                        className="lg:col-span-1"
                    >
                        <h3 className="text-2xl text-gold mb-1">Astras</h3>
                        <p className="font-serif italic text-text-muted text-sm mb-5">Skills, sharpened like divine weapons</p>
                        <div className="space-y-4 temple-card p-6 border-t-4 !border-t-saffron">
                            {skills.map((skill, index) => (
                                <motion.div
                                    key={index}
                                    whileHover="hover"
                                    initial="rest"
                                    className="group"
                                >
                                    <div className="flex justify-between mb-1">
                                        <span className="text-xs font-medium text-palm/90 group-hover:text-saffron transition-colors">{skill.name}</span>
                                        <span className="text-[10px] font-medium text-text-muted">{skill.level}</span>
                                    </div>
                                    <div className="w-full bg-ink rounded-full h-1.5 relative border border-gold/10">
                                        <motion.div
                                            className="bg-gradient-to-r from-kumkum via-saffron to-gold h-1.5 rounded-full shadow-[0_0_10px_rgba(240,138,36,0.5)] relative"
                                            initial={{ width: 0 }}
                                            whileInView={{ width: skill.level }}
                                            viewport={{ once: true }}
                                            transition={{ type: "spring", stiffness: 50, damping: 12, delay: 0.2 }}
                                            variants={agniVariant}
                                        >
                                            {/* Arrowhead at the tip of the bar */}
                                            <span className="absolute -right-1 top-1/2 -translate-y-1/2 w-0 h-0 border-y-[5px] border-y-transparent border-l-[8px] border-l-gold" aria-hidden="true" />
                                            <div className="absolute inset-0 w-full h-full overflow-visible">
                                                <div className="absolute w-1 h-1 bg-gold/70 rounded-full bottom-0 left-[15%] animate-ember-1 opacity-0 group-hover:opacity-100"></div>
                                                <div className="absolute w-1.5 h-1.5 bg-saffron/70 rounded-full bottom-0 left-[40%] animate-ember-2 opacity-0 group-hover:opacity-100"></div>
                                                <div className="absolute w-1 h-1 bg-gold/70 rounded-full bottom-0 left-[65%] animate-ember-3 opacity-0 group-hover:opacity-100"></div>
                                                <div className="absolute w-1 h-1 bg-kumkum/70 rounded-full bottom-0 left-[85%] animate-ember-1 opacity-0 group-hover:opacity-100"></div>
                                            </div>
                                        </motion.div>
                                    </div>
                                </motion.div>
                            ))}
                        </div>
                    </motion.div>

                    {/* Machine Learning — the Vidyas */}
                    <motion.div
                        initial={{ opacity: 0, x: 50 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.4 }}
                        className="lg:col-span-1"
                    >
                        <h3 className="text-2xl text-gold mb-1">Vidyā</h3>
                        <p className="font-serif italic text-text-muted text-sm mb-5">Machine learning knowledge</p>
                        <div className="flex flex-wrap gap-2 temple-card p-6 border-r-4 !border-r-kumkum">
                            {mlSkills.map((skill, index) => (
                                <motion.span
                                    key={index}
                                    initial={{ opacity: 0, scale: 0.8 }}
                                    whileInView={{ opacity: 1, scale: 1 }}
                                    viewport={{ once: true }}
                                    transition={{ delay: index * 0.05 + 0.5 }}
                                    className="px-3 py-1 bg-gold/10 text-gold border border-gold/25 rounded-full text-[11px] font-semibold hover:bg-saffron/20 hover:text-saffron transition-colors cursor-default"
                                >
                                    {skill}
                                </motion.span>
                            ))}
                        </div>
                    </motion.div>

                </div>
            </div>

            {/* Instagram Profile Gateway Modal */}
            <FullScreenOverlay
                isOpen={isInstagramOpen}
                onClose={() => setIsInstagramOpen(false)}
                title="📸 Instagram Gallery"
            >
                <div className="bg-ink-card border border-gold/25 rounded-2xl w-full max-w-lg overflow-hidden shadow-2xl relative">
                    {/* Instagram Header Bar */}
                    <div className="bg-ink p-4 flex items-center justify-between border-b border-gold/20">
                        <span className="font-bold text-gray-200">adarshnair01</span>
                        <div className="flex gap-4">
                            <div className="w-5 h-5 border-2 border-white/20 rounded-md"></div>
                            <div className="w-5 h-5 border-2 border-white/20 rounded-full"></div>
                        </div>
                    </div>

                    {/* Profile Stats Section */}
                    <div className="p-6 flex items-center gap-6">
                        {/* Profile Pic Ring */}
                        <div className="w-20 h-20 rounded-full p-[2px] bg-gradient-to-tr from-yellow-400 via-red-500 to-purple-500">
                            <div className="w-full h-full rounded-full bg-black p-0.5">
                                <img
                                    src="/an-logo.svg"
                                    alt="Profile"
                                    className="w-full h-full rounded-full object-cover bg-gray-800"
                                />
                            </div>
                        </div>

                        <div className="flex-1">
                            <div className="flex justify-between text-center mb-3 text-white">
                                <div><span className="font-bold block text-lg">50+</span><span className="text-xs text-gray-400">Posts</span></div>
                                <div><span className="font-bold block text-lg">1.2k</span><span className="text-xs text-gray-400">Followers</span></div>
                                <div><span className="font-bold block text-lg">800</span><span className="text-xs text-gray-400">Following</span></div>
                            </div>
                            <h4 className="font-bold text-white text-sm">Adarsh Nair</h4>
                            <p className="text-xs text-gray-400">Data Science | Travel | Photography 📸</p>
                        </div>
                    </div>

                    {/* Feed Preview (Using Interest Images as Mock Feed) */}
                    <div className={`grid ${INSTAGRAM_TOKEN ? 'grid-cols-3' : 'grid-cols-3'} gap-0.5 mb-6 bg-black`}>
                        {displayPosts.map((item, i) => (
                            <div
                                key={i}
                                className="aspect-square bg-gray-800 relative group overflow-hidden cursor-pointer"
                                onClick={() => window.open(item.permalink || 'https://www.instagram.com/adarshnair01/', '_blank')}
                            >
                                <img
                                    src={item.url}
                                    alt="Post"
                                    className="w-full h-full object-cover transition-transform group-hover:scale-110"
                                    onError={(e) => { e.target.onerror = null; e.target.src = 'https://via.placeholder.com/300?text=IG'; }} // Fallback if regular URL fails
                                />
                                <div className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2 text-white font-bold text-xs">
                                    <span>❤️ View</span>
                                </div>
                            </div>
                        ))}

                        {/* Placeholder Squares for typical 3xN grid look (Only show if MOCKING and need filler) */}
                        {showPlaceholders && [1, 2, 3].map((_, i) => (
                            <div key={i + 3} className="aspect-square bg-gray-900 flex items-center justify-center">
                                <span className="text-gray-700 text-xs">Post</span>
                            </div>
                        ))}
                    </div>

                    {/* CTA Button */}
                    <div className="p-6 pt-0">
                        <button
                            onClick={() => window.open('https://www.instagram.com/adarshnair01/', '_blank')}
                            className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-bold py-3 rounded-xl transition-all shadow-lg hover:shadow-purple-500/20 flex items-center justify-center gap-2"
                        >
                            <Camera size={18} /> View Full Profile
                        </button>
                        <p className="text-[10px] text-gray-500 text-center mt-3">
                            Click to open official Instagram page in a new tab.
                        </p>
                    </div>
                </div>
            </FullScreenOverlay>
        </section>
    );
};

export default About;
