import { motion } from 'framer-motion';
import { useState } from 'react';
import ProjectHUD from './ProjectHUD';
import { Github, Code } from 'lucide-react';

const Projects = () => {
    const [selectedProject, setSelectedProject] = useState(null);

    const projects = [
        {
            id: 'REC-SYS-01',
            title: 'Recommendation Engine',
            desc: 'Built a scalable recommendation engine for a leading restaurant brand to personalize customer experience and increase revenue.',
            technicalDesc: 'Architected and deployed a hybrid recommendation system leveraging collaborative filtering and content-based filtering algorithms. Processed over 10TB of transaction data to generate real-time personalized menu suggestions for 5M+ daily active users. The system uses a two-tower neural network architecture for candidate retrieval and ranking.',
            tags: ['Python', 'TensorFlow', 'Redis', 'AWS SageMaker', 'Kubernetes'],
            latency: '< 50ms',
            accuracy: '95%',
            features: [
                'Real-time inference < 50ms latency',
                'Hybrid Two-Tower Neural Network Architecture',
                'A/B Testing Framework Integration',
                'Automated Model Retraining Pipeline (Airflow)',
                'Scalable to 10M+ users'
            ],
            challenge: 'Handling the "Cold Start" problem for new users while maintaining sub-50ms inference time during peak lunch hours.',
            solution: 'Implemented a fallback popularity-based heuristic for new users and used Redis caching for pre-computed embeddings of top items, reducing P99 latency by 40%.',
            metrics: [
                { value: '+15%', label: 'Revenue Lift' },
                { value: '50ms', label: 'Avg Latency' },
                { value: '5M+', label: 'Daily Users' }
            ],
            link: 'https://github.com/adarshnair01/Recommendation-Engine',
            image: '/recommendation-engine.png'
        },
        {
            id: 'TS-FC-02',
            title: 'Sales Forecasting System',
            desc: 'End-to-end development of a sales forecasting system improving accuracy by 25% using time-series modeling.',
            technicalDesc: 'Developed a robust multi-variate time-series forecasting pipeline using Facebook Prophet and LSTM networks. The system predicts inventory requirements for 2,000+ store locations up to 4 weeks in advance, accounting for seasonality, holidays, and local events.',
            tags: ['Time Series', 'ARIMA', 'Prophet', 'Python', 'Azure Data Factory'],
            latency: '< 15ms',
            accuracy: '92%',
            features: [
                'Multi-horizon forecasting (Daily, Weekly)',
                'Automated Anomaly Detection',
                'Integration with Supply Chain ERP',
                'Event-based lift quantization',
                'Dashboard for Regional Managers'
            ],
            challenge: 'High variance in sales data due to localized events/promotions made standard ARIMA models inaccurate.',
            solution: 'Integrated external regressors (weather, promotions, holidays) into the Prophet model and ensemble techniques, boosting forecast accuracy (MAPE) from 78% to 92%.',
            metrics: [
                { value: '+25%', label: 'Accuracy' },
                { value: '$2M', label: 'Waste Saved' },
                { value: '2k+', label: 'Stores' }
            ],
            link: '#',
            image: '/sales-forecasting.png'
        },
        {
            id: 'CHURN-PRED-03',
            title: 'Customer Churn Prediction',
            desc: 'Implemented predictive models to identify at-risk customers and developed segmentation strategies.',
            technicalDesc: 'Engineered a churn prediction classifier using XGBoost and Random Forest on a dataset of 2M+ customers. Identified key behavioral markers (e.g., reduced session frequency, support ticket volume) to trigger automated retention campaigns.',
            tags: ['Scikit-learn', 'XGBoost', 'Pandas', 'Tableau', 'SQL'],
            latency: '< 10ms',
            accuracy: '81%',
            features: [
                'Propensity Score Modeling',
                'Dynamic Customer Segmentation (RFM)',
                'Automated Email Trigger Integration',
                'Explainable AI (SHAP Values)',
                'Monthly Cohort Analysis'
            ],
            challenge: 'Imbalanced dataset (only 5% churn rate) causing models to bias towards the majority class.',
            solution: 'Applied SMOTE (Synthetic Minority Over-sampling Technique) and calibrated probability thresholds to optimize Recall without sacrificing Precision.',
            metrics: [
                { value: '85%', label: 'Recall' },
                { value: '-10%', label: 'Churn Rate' },
                { value: '1.2M', label: 'Retained' }
            ],
            link: '#',
            image: '/churn-prediction.png'
        }
    ];

    return (
        <section id="projects" className="py-10 bg-transparent">
            {/* Tech Background decoration */}
            <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-[-1]">
                <div className="absolute top-[20%] right-[10%] w-64 h-64 bg-primary/5 rounded-full blur-[100px]"></div>
                <div className="absolute bottom-[20%] left-[10%] w-64 h-64 bg-secondary/5 rounded-full blur-[100px]"></div>
            </div>

            <div className="container mx-auto px-4">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="text-center mb-16"
                >
                    <h2 className="text-3xl md:text-4xl font-bold text-white mb-4 font-harry tracking-wider">Featured Projects</h2>
                    <p className="text-gray-400 font-mono text-xs tracking-widest uppercase opacity-70">
                        Select a module to view system architecture
                    </p>
                </motion.div>

                {/* Technical Projects */}
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {projects.map((project, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 30 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: index * 0.1 }}
                            onClick={() => setSelectedProject(project)}
                            className="group bg-[#1e1e1e] rounded-xl overflow-hidden border border-gray-800 hover:border-[#00f3ff] transition-all duration-300 hover:shadow-[0_0_20px_rgba(0,243,255,0.2)] hover:-translate-y-1 cursor-pointer relative"
                        >
                            {/* Card Scanline Effect */}
                            <div className="absolute inset-0 bg-gradient-to-b from-[#00f3ff]/0 via-[#00f3ff]/5 to-[#00f3ff]/0 translate-y-[-100%] group-hover:translate-y-[100%] transition-transform duration-1000 pointer-events-none z-10"></div>

                            <div className="h-48 w-full overflow-hidden relative">
                                <img
                                    src={project.image}
                                    alt={project.title}
                                    className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110 opacity-80 group-hover:opacity-100 group-hover:grayscale-0 grayscale"
                                />
                                <div className="absolute inset-0 bg-gradient-to-t from-[#1e1e1e] to-transparent opacity-90"></div>

                                {/* Corner Decorations */}
                                <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <Code className="text-[#00f3ff]" size={20} />
                                </div>
                            </div>

                            <div className="p-6 relative z-20">
                                <div className="text-[10px] text-[#00f3ff] font-mono mb-2 opacity-0 group-hover:opacity-100 transition-opacity tracking-widest uppercase">
                                    Click to Initialize
                                </div>
                                <h3 className="text-xl font-bold text-gray-200 mb-2 group-hover:text-[#00f3ff] transition-colors font-harry tracking-wide">{project.title}</h3>
                                <p className="text-gray-400 mb-4 text-sm line-clamp-3 leading-relaxed font-sans">{project.desc}</p>
                                <div className="flex flex-wrap gap-2">
                                    {project.tags.slice(0, 3).map(tag => (
                                        <span key={tag} className="px-2 py-1 text-[10px] font-bold text-gray-400 bg-gray-800 rounded group-hover:text-[#00f3ff] group-hover:bg-[#00f3ff]/10 border border-transparent group-hover:border-[#00f3ff]/30 transition-all font-mono">
                                            {tag}
                                        </span>
                                    ))}
                                    {project.tags.length > 3 && (
                                        <span className="px-2 py-1 text-[10px] font-bold text-gray-500 font-mono">+{project.tags.length - 3}</span>
                                    )}
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>

            {/* IRON MAN / JARVIS STYLE MODAL */}
            <ProjectHUD
                isOpen={!!selectedProject}
                onClose={() => setSelectedProject(null)}
                project={selectedProject}
                onNext={() => {
                    const currentIndex = projects.findIndex(p => p.id === selectedProject?.id);
                    if (currentIndex !== -1) {
                        setSelectedProject(projects[(currentIndex + 1) % projects.length]);
                    }
                }}
                onPrev={() => {
                    const currentIndex = projects.findIndex(p => p.id === selectedProject?.id);
                    if (currentIndex !== -1) {
                        setSelectedProject(projects[(currentIndex - 1 + projects.length) % projects.length]);
                    }
                }}
            />

        </section>
    );
};

export default Projects;
