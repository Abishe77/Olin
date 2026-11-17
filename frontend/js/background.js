
const canvas = document.getElementById("background-dots");
const ctx = canvas.getContext("2d");

const spacing = 20;
const dotRadius = 1;

const influenceRadius = 100;
const distortStrength = 35;

const dots = [];

const pointer = {
    x: null,
    y: null
};

window.addEventListener("pointermove", (event) => {
    pointer.x = event.clientX;
    pointer.y = event.clientY;
});

window.addEventListener("pointerleave", () => {
    pointer.x = null;
    pointer.y = null;
});


function resizeCanvas() {

    const width = window.innerWidth;
    const height = window.innerHeight;

    const pixelRatio = window.devicePixelRatio || 1;

    canvas.width = width * pixelRatio;
    canvas.height = height * pixelRatio;

    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

    dots.length = 0;

    for (let x = 0; x < width; x += spacing) {

        for (let y = 0; y < height; y += spacing) {

            dots.push({
                originalX: x,
                originalY: y,
                x: x,
                y: y
            });

        }
    }
}


function drawDots(width, height) {

    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = "rgba(255, 255, 255, 0.22)";

    for (const dot of dots) {

        if (pointer.x !== null && pointer.y !== null) {

            const distance = getDistance(
                dot.originalX,
                dot.originalY,
                pointer.x,
                pointer.y
            );

            const influence = getInfluence(distance, influenceRadius);

            const dx = dot.originalX - pointer.x;
            const dy = dot.originalY - pointer.y;

            if (distance === 0) {
                dot.x += (dot.originalX - dot.x) * 0.18;
                dot.y += (dot.originalY - dot.y) * 0.18;
                continue;
            }

            const targetX = dot.originalX + (dx / distance) * influence * distortStrength;
            const targetY = dot.originalY + (dy / distance) * influence * distortStrength;

            dot.x += (targetX - dot.x) * 0.18;
            dot.y += (targetY - dot.y) * 0.18;

        } else {

            dot.x += (dot.originalX - dot.x) * 0.18;
            dot.y += (dot.originalY - dot.y) * 0.18;

        }

        ctx.beginPath();
        ctx.arc(dot.x, dot.y, dotRadius, 0, Math.PI * 2);
        ctx.fill();
    }
}


function getDistance(x1, y1, x2, y2) {

    const dx = x2 - x1;
    const dy = y2 - y1;

    return Math.sqrt(dx * dx + dy * dy);
}


function getInfluence(distance, radius) {

    return Math.exp(
        -(distance * distance) / (2 * radius * radius)
    );
}


function animate() {

    drawDots(window.innerWidth, window.innerHeight);

    requestAnimationFrame(animate);
}


window.addEventListener("resize", resizeCanvas);

resizeCanvas();
animate();
