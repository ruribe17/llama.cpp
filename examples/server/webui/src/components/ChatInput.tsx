import { useState } from "react";
import TranslateButton from "./TranslateButton";

const ChatInput = () => {
    const [inputText, setInputText] = useState("");

    const handleSendMessage = () => {
        if (!inputText.trim()) return;
        alert(`전송된 메시지: ${inputText}`); // 실제 메시지 전송 기능과 연결 필요
        setInputText("");
    };

    return (
        <div className="flex items-center space-x-2 p-4">
            <input
                type="text"
                className="border p-2 flex-grow rounded-md"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="메시지를 입력하세요..."
            />
            <TranslateButton inputText={inputText} setInputText={setInputText} />
            <button
                className="bg-blue-500 text-white px-3 py-1 rounded-md"
                onClick={handleSendMessage}
            >
                전송
            </button>
        </div>
    );
};

export default ChatInput;
