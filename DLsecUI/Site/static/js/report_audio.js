

// 假设你的数据是这样的
var data = [
    { algorithm: 'FGSM', successRate: 0, averageP: 8.5593, totalScore: 91.5914 },
    { algorithm: 'PGD', successRate: 0.8, averageP: 3.1502, totalScore: 28.5762 },
    { algorithm: 'CW', successRate: 0.8, averageP: 2.4076, totalScore: 26.5813 }
];

// 创建第一个图表
const chart1 = new G2.Chart({
    container: 'container1',
    autoFit: true,
    height: 300,
    width:350
});

chart1.data(data);

chart1.scale('successRate', {
    min: 0,
    max: 1,
});
chart1.axis('successRate', {
    title: {},
});
chart1.tooltip({
    showMarkers: false,
    shared: true,
});

chart1.interval().position('algorithm*successRate');

chart1.render();

// 创建第二个图表
const chart2 = new G2.Chart({
    container: 'container2',
    autoFit: true,
    height: 300,
    width:350
});

chart2.data(data);

chart2.scale('averageP', {
    min: 0,
    max: 10,
});
chart2.axis('averageP', {
    title: {},
});
chart2.tooltip({
    showMarkers: false,
    shared: true,
});

chart2.interval().position('algorithm*averageP');

chart2.render();

// var FGSM_score = data.find(item => item.algorithm === 'FGSM').totalScore;
// var PGD_score = data.find(item => item.algorithm === 'PGD').totalScore;
// var CW_score = data.find(item => item.algorithm === 'CW').totalScore;

// var chart = [];
// // 添加到雷达图的数据中
// chart.push({ item: "FGSM", score: FGSM_score });
// chart.push({ item: "PGD", score: PGD_score });
// chart.push({ item: "CW", score: CW_score });

// // 创建雷达图
// const chart_final = new G2.Chart({
//     container: 'final-report-1',
//     autoFit: true,
//     width: 400,
//     height: 400,
//     padding: [20, 20, 95, 20]
// });

// chart_final.data(chart);
// chart_final.scale('score', {
//     min: 0,
//     max: 100
// });
// chart_final.coordinate('polar', {
//     radius: 0.8
// });
// chart_final.tooltip({
//     shared: true,
//     showCrosshairs: true,
//     crosshairs: {
//         line: {
//             style: {
//                 lineDash: [4, 4],
//                 stroke: '#333'
//             }
//         }
//     }
// });

// chart_final.axis('item', {
//     line: null,
//     tickLine: null,
//     grid: {
//         line: {
//             style: {
//                 lineDash: null
//             }
//         }
//     }
// });
// chart_final.axis('score', {
//     line: null,
//     tickLine: null,
//     grid: {
//         type: 'polygon',
//         line: {
//             style: {
//                 lineDash: null
//             }
//         }
//     }
// });

// chart_final.legend('user', {
//     marker: 'circle'
// });

// chart_final
//     .area()
//     .position('item*score')
//     .color('orange', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
//     .style({ opacity: 0.5 });

// chart_final
//     .line()
//     .position('item*score')
//     .color('orange')
//     .size(2)
//     .style({
//         fillOpacity: 0.2
//     });

// chart_final
//     .point()
//     .position('item*score')
//     .color('user')
//     .shape('circle')
//     .size(4)
//     .style({
//         stroke: '#fff',
//         lineWidth: 1,
//         fillOpacity: 1
//     });

// chart_final.render();

var totalScore = data.reduce((sum, item) => sum + item.totalScore, 0) / data.length;

// 获取 final-score 元素
var finalScoreElement = document.getElementById("final-score");

// 显示总得分
finalScoreElement.innerHTML = `
    <p><strong>总得分:</strong> ${totalScore}</p>
`;

// 获取 result 元素
var resultElement = document.getElementById("result");

// 创建一个新的 div 元素来包含所有的 report-box 元素
var reportBoxesElement = document.createElement("div");
reportBoxesElement.classList.add("report-boxes");

// 显示每个算法的测试结果
data.map(item => {
    // 创建一个新的 div 元素
    var element = document.createElement("div");
    element.classList.add("report-box");

    element.innerHTML = `
        <h3 style="font-size: 1.5rem; font-weight:bold; color: var(--blue);">${item.algorithm}测试方法相关结果</h3>
        <p><strong>${item.algorithm} 攻击成功率:</strong> ${item.successRate}</p>
        <p><strong>${item.algorithm} 平均P范数距离:</strong> ${item.averageP}</p>
        <p><strong>${item.algorithm} 总分:</strong> ${item.totalScore}</p>
    `;

    // 将新的 div 元素添加到 report-boxes 元素中
    reportBoxesElement.appendChild(element);
});

// 将 report-boxes 元素添加到 result 元素中
resultElement.appendChild(reportBoxesElement);