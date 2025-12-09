import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import { Camera, MapPin, Smile } from 'lucide-react';
import FullScreenOverlay from './FullScreenOverlay';

// Custom variant for the boiling potion effect
const potionVariant = {
    rest: { scale: 1, filter: "brightness(1)" },
    hover: {
        scale: 1.02,
        filter: "brightness(1.2)",
        backgroundColor: "#22c55e", // Green color
        transition: {
            duration: 0.3,
            yoyo: Infinity
        }
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
            desc: 'Enjoying sci-fi novels and technical blogs.',
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
        <section id="about" className="py-20 bg-transparent relative">
            <div className="container mx-auto px-4">

                {/* Section Header */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="text-center mb-16"
                >
                    <h2 className="text-3xl md:text-4xl font-bold text-white mb-4 font-harry tracking-wider">About Me</h2>
                </motion.div>

                <div className="grid grid-cols-1 lg:grid-cols-4 gap-8 items-start">

                    {/* Bio & Interests - 50% width */}
                    <motion.div
                        initial={{ opacity: 0, x: -50 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.2 }}
                        className="lg:col-span-2"
                    >
                        <h3 className="text-2xl font-semibold text-gray-200 mb-6 font-harry">Who I Am</h3>
                        <p className="text-gray-400 mb-8 leading-relaxed bg-[#1e1e1e] p-6 rounded-md shadow-lg border-l-4 border-primary">
                            I am a data science professional from Bengaluru, India, with a passion for extracting actionable insights from complex datasets.
                            My expertise spans across machine learning, predictive modeling, and software architecture.
                            I thrive on solving challenging problems, from improving sales forecasting to building large-scale recommendation systems.
                        </p>

                        {/* Personal Interests (Replaced Games) */}
                        <div className="mt-8">
                            <h4 className="text-lg font-semibold text-gray-300 mb-4 flex items-center gap-2">
                                <span className="text-secondary">🌟</span> Personal Interests
                            </h4>
                            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                                {interests.map((item, index) => (
                                    <motion.div
                                        key={index}
                                        whileHover={{ scale: 1.02 }}
                                        onClick={() => setIsInstagramOpen(true)}
                                        className="group relative bg-[#1e1e1e] rounded-xl overflow-hidden shadow-md border border-black hover:shadow-xl transition-all h-[100px] cursor-pointer"
                                    >
                                        <div
                                            className="absolute inset-0 bg-cover bg-center transition-transform duration-700 group-hover:scale-110 opacity-40 group-hover:opacity-60"
                                            style={{ backgroundImage: `url(${item.url})` }}
                                        ></div>
                                        <div className="absolute inset-0 bg-gradient-to-t from-black via-black/50 to-transparent"></div>

                                        <div className="absolute inset-0 p-3 flex flex-col justify-end">
                                            <div className="flex items-center gap-2">
                                                <item.icon size={14} className="text-white drop-shadow-md" />
                                                <h5 className="font-bold text-gray-100 text-[10px] lg:text-xs tracking-wide font-harry shadow-black drop-shadow-md leading-tight">{item.title}</h5>
                                            </div>
                                        </div>
                                    </motion.div>
                                ))}
                            </div>
                        </div>
                    </motion.div>

                    {/* Skills - 25% width */}
                    <motion.div
                        initial={{ opacity: 0, y: 30 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.3 }}
                        className="lg:col-span-1"
                    >
                        <h3 className="text-2xl font-semibold text-gray-200 mb-6 font-harry">Skills</h3>
                        <div className="space-y-4 bg-[#1e1e1e] p-6 rounded-md shadow-lg border-t-4 border-secondary">
                            {skills.map((skill, index) => (
                                <motion.div
                                    key={index}
                                    whileHover="hover"
                                    initial="rest"
                                    className="group cursor-pointer"
                                >
                                    <div className="flex justify-between mb-1">
                                        <span className="text-xs font-medium text-gray-300 group-hover:text-primary transition-colors">{skill.name}</span>
                                        <span className="text-[10px] font-medium text-gray-500">{skill.level}</span>
                                    </div>
                                    <div className="w-full bg-gray-700 rounded-full h-1.5 overflow-hidden relative">
                                        <motion.div
                                            className="bg-white h-1.5 rounded-full shadow-[0_0_10px_rgba(255,255,255,0.5)] relative overflow-hidden"
                                            initial={{ width: 0 }}
                                            whileInView={{ width: skill.level }}
                                            viewport={{ once: true }}
                                            transition={{ type: "spring", stiffness: 50, damping: 10, delay: 0.2 }}
                                            variants={potionVariant}
                                        >
                                            <div className="absolute inset-0 w-full h-full">
                                                <div className="bubble absolute w-1 h-1 bg-white/40 rounded-full bottom-0 left-[10%] animate-bubble-1 opacity-0 group-hover:opacity-100"></div>
                                                <div className="bubble absolute w-1.5 h-1.5 bg-white/30 rounded-full bottom-0 left-[30%] animate-bubble-2 opacity-0 group-hover:opacity-100 delay-100"></div>
                                                <div className="bubble absolute w-1 h-1 bg-white/40 rounded-full bottom-0 left-[50%] animate-bubble-3 opacity-0 group-hover:opacity-100 delay-200"></div>
                                                <div className="bubble absolute w-2 h-2 bg-white/30 rounded-full bottom-0 left-[70%] animate-bubble-1 opacity-0 group-hover:opacity-100 delay-300"></div>
                                                <div className="bubble absolute w-1 h-1 bg-white/40 rounded-full bottom-0 left-[90%] animate-bubble-2 opacity-0 group-hover:opacity-100 delay-100"></div>
                                            </div>
                                        </motion.div>
                                    </div>
                                </motion.div>
                            ))}
                        </div>
                    </motion.div>

                    {/* Machine Learning & Algorithms - 25% width */}
                    <motion.div
                        initial={{ opacity: 0, x: 50 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.4 }}
                        className="lg:col-span-1"
                    >
                        <h3 className="text-2xl font-semibold text-gray-200 mb-6 font-harry">Machine Learning</h3>
                        <div className="flex flex-wrap gap-2 bg-[#1e1e1e] p-6 rounded-md shadow-lg border-r-4 border-primary">
                            {mlSkills.map((skill, index) => (
                                <motion.span
                                    key={index}
                                    initial={{ opacity: 0, scale: 0.8 }}
                                    whileInView={{ opacity: 1, scale: 1 }}
                                    viewport={{ once: true }}
                                    transition={{ delay: index * 0.05 + 0.5 }}
                                    className="px-3 py-1 bg-secondary/10 text-secondary border border-secondary/20 rounded-full text-[10px] font-bold hover:bg-secondary/20 transition-colors cursor-default"
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
                <div className="bg-[#1e1e1e] border border-gray-800 rounded-2xl w-full max-w-lg overflow-hidden shadow-2xl relative">
                    {/* Instagram Header Bar */}
                    <div className="bg-[#121212] p-4 flex items-center justify-between border-b border-gray-800">
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
                                    src="/itachi.png"
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
