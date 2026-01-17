import './Feedback.css';

interface LivesDisplayProps {
    lives: number;
    maxLives?: number;
}

export default function LivesDisplay({ lives, maxLives = 3 }: LivesDisplayProps) {
    return (
        <div className="lives-display">
            {Array.from({ length: maxLives }).map((_, i) => (
                <span
                    key={i}
                    className={`life-heart ${i < lives ? '' : 'life-lost'}`}
                >
                    ❤️
                </span>
            ))}
        </div>
    );
}
