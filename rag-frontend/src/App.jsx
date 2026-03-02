import { useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { Send } from "lucide-react";

function App() {
    const [question, setQuestion] = useState("");
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);

    const sendMessage = async () => {
        if (!question.trim()) return;

        const userMessage = { role: "user", content: question };
        setMessages((prev) => [...prev, userMessage]);
        setLoading(true);

        try {
            const res = await axios.post("http://127.0.0.1:8000/query", {
                question: question,
            });

            const botMessage = {
                role: "assistant",
                content: res.data.answer,
                sources: res.data.sources || [],
            };

            setMessages((prev) => [...prev, botMessage]);
        } catch (error) {
            console.error("Error fetching response:", error);
        }

        setLoading(false);
        setQuestion("");
    };

    return (
        <div className="app">
            <h1 className="title">RAG Assistant</h1>

            <div className="chat-container">
                {messages.map((msg, index) => (
                    <motion.div
                        key={index}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={`message ${msg.role}`}
                    >
                        <p>{msg.content}</p>

                        {msg.sources && msg.sources.length > 0 && (
                            <div className="sources">
                                <strong>Sources:</strong>
                                {msg.sources.map((src, i) => (
                                    <div key={i} className="source-card">
                                        <p><strong>Source:</strong> {src.source}</p>
                                        <p><strong>Page:</strong> {src.page}</p>
                                    </div>
                                ))}
                            </div>
                        )}
                    </motion.div>
                ))}

                {loading && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="message assistant"
                    >
                        <p>Thinking...</p>
                    </motion.div>
                )}
            </div>

            <div className="input-container">
                <input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Ask something..."
                    onKeyDown={(e) => {
                        if (e.key === "Enter") sendMessage();
                    }}
                />
                <button onClick={sendMessage} disabled={loading}>
                    <Send size={18} />
                </button>
            </div>
        </div>
    );
}

export default App;