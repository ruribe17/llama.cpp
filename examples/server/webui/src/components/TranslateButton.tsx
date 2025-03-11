import { useState } from "react";

interface TranslateButtonProps {
    inputText: string;
    setInputText: (text: string) => void;
}

const TranslateButton: React.FC<TranslateButtonProps> = ({ inputText, setInputText }) => {
    const [loading, setLoading] = useState(false);

    const handleTranslate = async () => {
        if (!inputText.trim()) return;

        setLoading(true);
        try {
            const response = await fetch(
                `https://api.mymemory.translated.net/get?q=${encodeURIComponent(inputText)}&langpair=ko|en`
            );
            const data = await response.json();
            if (data.responseData && data.responseData.translatedText) {
                setInputText(data.responseData.translatedText);
            } else {
                alert("번역 실패");
            }
        } catch (error) {
            console.error("Translation Error:", error);
            alert("번역 요청 중 오류 발생!");
        } finally {
            setLoading(false);
        }
    };

    return (
        <button
            className="bg-green-500 text-white px-3 py-1 rounded-md"
            onClick={handleTranslate}
            disabled={loading}
        >
            {loading ? "번역 중..." : "번역"}
        </button>
    );
};

export default TranslateButton;
