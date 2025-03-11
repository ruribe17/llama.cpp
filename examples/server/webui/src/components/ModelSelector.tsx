import { useState } from "react";

const ModelSelector = () => {
    console.log("ModelSelector 렌더링됨"); // 디버깅 로그 추가
    const [selectedModel, setSelectedModel] = useState("default");

    const handleChangeModel = async () => {
        try {
            const response = await fetch("/change_model", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model: selectedModel }),
            });

            const data = await response.json();

            if (data.success) {
                alert(`모델이 성공적으로 변경되었습니다: ${selectedModel}`);
            } else {
                alert(`모델 변경 실패: ${data.error}`);
            }
        } catch (error) {
            console.error("Error:", error);
            alert("서버 요청 중 오류 발생!");
        }
    };

    return (
        <div className="flex items-center space-x-2 p-2">
            <select
                className="border p-2 rounded-md"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
            >
                <option value="default">기본 모델</option>
                <option value="coder">코딩 특화 모델</option>
                <option value="korean">한국어 모델</option>
                <option value="light">경량 모델</option>
            </select>
            <button
                className="bg-blue-500 text-white px-3 py-1 rounded-md"
                onClick={handleChangeModel}
            >
                모델 변경
            </button>
        </div>
    );
};

export default ModelSelector;
