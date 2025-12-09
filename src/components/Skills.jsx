import { motion } from 'framer-motion';

const Skills = () => {
    const technicalSkills = [
        { name: 'Python/Julia', level: 95 },
        { name: 'SQL', level: 75 },
        { name: 'ElasticSearch', level: 85 },
        { name: 'C/C++/C#', level: 70 },
    ];

    const mlSkills = [
        'Regression (Linear, Logistic)', 'Kalman Filters', 'Decision Tree',
        'Support Vector Machine', 'K-Nearest Neighbors', 'Random Forest',
        'Gradient Boosting', 'AdaBoost', 'PCA', 'Neural Networks', 'Optimization'
    ];

    const tools = [
        'Visual Studio', 'PyCharm', 'Atom', 'Eclipse', 'SQL Management Studio', 'Git'
    ];

    const languages = [
        'English', 'Hindi', 'Malayalam', 'Gujarati'
    ];

    return (
        <section id="skills" className="py-20 bg-transparent">
            <div className="container mx-auto px-4">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="text-center mb-16"
                >
                    <h2 className="text-3xl md:text-4xl font-bold text-white mb-4 font-harry tracking-wider">Skills & Expertise</h2>
                </motion.div>

                <div className="grid md:grid-cols-2 gap-16">
                    {/* Technical Skills - Progress Bars */}
                    <div>
                        <h3 className="text-xl font-bold text-gray-200 mb-8 border-l-4 border-primary pl-4 font-harry tracking-wide">Technical Proficiency</h3>
                        <div className="space-y-6 bg-[#1e1e1e] p-6 rounded-2xl border border-black shadow-lg backdrop-blur-sm">
                            {technicalSkills.map((skill, index) => (
                                <div key={index}>
                                    <div className="flex justify-between mb-2">
                                        <span className="text-gray-300 font-medium">{skill.name}</span>
                                        <span className="text-gray-500 text-sm font-bold">{skill.level}%</span>
                                    </div>
                                    <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                                        <motion.div
                                            className="h-full bg-primary"
                                            initial={{ width: 0 }}
                                            whileInView={{ width: `${skill.level}%` }}
                                            viewport={{ once: true }}
                                            transition={{ duration: 1, delay: 0.2 }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>

                        <h3 className="text-xl font-bold text-gray-200 mb-6 mt-12 border-l-4 border-secondary pl-4 font-harry tracking-wide">Languages</h3>
                        <div className="flex flex-wrap gap-3">
                            {languages.map((lang, i) => (
                                <span key={i} className="px-4 py-2 bg-[#1e1e1e] border border-black rounded-lg text-gray-300 text-sm font-medium shadow-sm hover:border-primary transition-colors">
                                    {lang}
                                </span>
                            ))}
                        </div>
                    </div>

                    {/* ML & Tools - Tags */}
                    <div>
                        <h3 className="text-xl font-bold text-gray-200 mb-8 border-l-4 border-secondary pl-4 font-harry tracking-wide">Machine Learning & Algorithms</h3>
                        <div className="flex flex-wrap gap-3 mb-10">
                            {mlSkills.map((skill, index) => (
                                <motion.span
                                    key={index}
                                    initial={{ opacity: 0, scale: 0.8 }}
                                    whileInView={{ opacity: 1, scale: 1 }}
                                    viewport={{ once: true }}
                                    transition={{ delay: index * 0.05 }}
                                    className="px-4 py-2 bg-secondary/10 text-secondary border border-secondary/20 rounded-full text-sm font-bold hover:bg-secondary/20 transition-colors cursor-default"
                                >
                                    {skill}
                                </motion.span>
                            ))}
                        </div>

                        <h3 className="text-xl font-bold text-gray-200 mb-6 border-l-4 border-primary pl-4 font-harry tracking-wide">Tools & IDEs</h3>
                        <div className="flex flex-wrap gap-3">
                            {tools.map((tool, index) => (
                                <span key={index} className="px-4 py-2 bg-[#1e1e1e] text-gray-300 rounded-md text-sm border border-black shadow-sm font-medium hover:border-primary transition-colors">
                                    {tool}
                                </span>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default Skills;
