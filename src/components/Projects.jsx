import { motion } from 'framer-motion';
import { useState } from 'react';
import ProjectHUD from './ProjectHUD';
import { Gift } from 'lucide-react';
import { FestiveHeading } from './Festive';

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
        <section id="projects" className="py-24 bg-transparent relative">
            <div className="container mx-auto px-4">
                <FestiveHeading
                    eyebrow="Chapter Three"
                    title="Featured Projects"
                    script="presents under the tree"
                    subtitle="Pick a gift to unwrap the full story"
                />

                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-10">
                    {projects.map((project, index) => {
                        const ribbon = ['bg-holly', 'bg-gold', 'bg-pine'][index % 3];
                        return (
                            <motion.button
                                type="button"
                                key={project.id}
                                initial={{ opacity: 0, y: 30 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: index * 0.1 }}
                                onClick={() => setSelectedProject(project)}
                                className="group text-left bg-night-card rounded-2xl border border-frost/15 hover:border-gold transition-all duration-300 hover:shadow-[0_0_35px_rgba(240,195,78,0.25)] hover:-translate-y-1 relative snow-cap"
                            >
                                {/* Bow on top of the gift */}
                                <span className="absolute -top-5 left-1/2 -translate-x-1/2 z-20 text-3xl group-hover:scale-125 group-hover:rotate-12 transition-transform" aria-hidden="true">🎀</span>

                                <div className="h-52 w-full overflow-hidden relative rounded-t-2xl">
                                    <img
                                        src={project.image}
                                        alt={project.title}
                                        className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110 opacity-85 group-hover:opacity-100"
                                    />
                                    <div className="absolute inset-0 bg-gradient-to-t from-night-card via-night-card/20 to-transparent"></div>
                                    {/* Ribbon across the box */}
                                    <span className={`absolute inset-y-0 left-1/2 -translate-x-1/2 w-4 ${ribbon} opacity-80 group-hover:opacity-0 transition-opacity duration-500`} aria-hidden="true" />
                                </div>

                                <div className="p-6 relative">
                                    <div className="text-[11px] text-gold mb-2 font-bold tracking-widest uppercase flex items-center gap-2">
                                        <Gift size={12} /> Gift No. {index + 1}
                                    </div>
                                    <h3 className="text-2xl font-bold text-snow mb-2 group-hover:text-gold transition-colors">{project.title}</h3>
                                    <p className="text-text-muted mb-4 text-sm line-clamp-3 leading-relaxed">{project.desc}</p>
                                    <div className="flex flex-wrap gap-2">
                                        {project.tags.slice(0, 3).map(tag => (
                                            <span key={tag} className="px-2 py-1 text-[10px] font-bold text-frost bg-frost/10 rounded-full border border-frost/20 group-hover:border-gold/40 group-hover:text-gold transition-all">
                                                {tag}
                                            </span>
                                        ))}
                                        {project.tags.length > 3 && (
                                            <span className="px-2 py-1 text-[10px] font-bold text-text-muted">+{project.tags.length - 3}</span>
                                        )}
                                    </div>
                                    <div className="mt-5 text-sm text-holly group-hover:text-[#ff8a99] transition-colors font-bold">
                                        Unwrap →
                                    </div>
                                </div>
                            </motion.button>
                        );
                    })}
                </div>
            </div>

            {/* Holiday letter modal */}
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
