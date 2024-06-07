
result={"Model": "ResNet56(Backdoored)", "CACC": "99.97", "ASR": "73.9", "NTE": "86.17", "ALDp": "82.92", "RGB": "75.09", "RIC": "67.08", "Tstd": "54.2", "Tsize": "49.2", "Score": "73.79"};
var result = JSON.parse(JSON.stringify(result));


// 创建雷达图
const chart_backdoor = new G2.Chart({
    container: 'final-report',
    autoFit: true,
    width:500,
    height: 500,
    padding: [20, 20, 95, 20]
    });
    
var chartData = Object.keys(result).map(key => {
    if (key !== "Model" && key !== "Score"){
        var score=parseFloat(result[key]);
        return {
            item:key,
            score:score
        }
    } 
}).filter(Boolean);
var score=(result["Score"]);


console.log(chartData);
chart_backdoor.data(chartData);
chart_backdoor.scale('score',{
    min:0,
    max:100
});
chart_backdoor.coordinate('polar',{
    radius:0.8
});
chart_backdoor.tooltip({
    shared:true,
    showCrosshairs:true,
    crosshairs:{
        line:{
            style:{
                lineDash:[4,4],
                stroke:'#333'
            }
        }
    }
});

chart_backdoor.axis('item', {
line: null,
tickLine: null,
grid: {
    line: {
    style: {
        lineDash: null
    }
    }
}
});
chart_backdoor.axis('score', {
line: null,
tickLine: null,
grid: {
    type: 'polygon',
    line: {
    style: {
        lineDash: null
    }
    }
}
});

chart_backdoor.legend('user', {
marker: 'circle'
});
chart_backdoor
    .area()
    .position('item*score')
    .color('lightblue', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    .style({ opacity: 0.5 });
chart_backdoor
.line()
.position('item*score')
.color('lightblue')
.size(2);

chart_backdoor
.point()
.position('item*score')
.color('user')
.shape('circle')
.size(4)
.style({
    stroke: '#fff',
    lineWidth: 1,
    fillOpacity: 1
});

chart_backdoor.render();

var element = document.getElementById("backdoor-score");

element.innerHTML = `
    <p><strong>Tstd评分:</strong> ${result["Tstd"]}</p>
    <p><strong>Tsize评分:</strong> ${result["Tsize"]}</p>
    <p><strong>后门评分:</strong> ${(0.1541*result["Tstd"]+0.1375*result['Tsize'])/0.2916}</p>
`;

element.style.fontSize = "1rem";
element.style.marginTop = "1rem";
element.style.display = "flex";
element.style.flexDirection = "column";
element.style.justifyContent = "center";
element.style.alignItems = "center";
element.style.textAlign = "center";
element.style.fontWeight = "bold";

var element=document.getElementById("adver-score");
element.innerHTML = `
    <p><strong>CACC评分:</strong> ${result["NTE"]}</p>
    <p><strong>ASR评分:</strong> ${result["RGB"]}</p>
    <p><strong>ALDp分:</strong> ${result["RIC"]}</p>
    <p><strong>对抗评分:</strong> ${((0.0752*result["NTE"]+0.0947*result['RGB']+0.083*result['RIC'])/0.2582)}</p>
`;

element.style.fontSize = "1rem";
element.style.marginTop = "1rem";
element.style.display = "flex";
element.style.flexDirection = "column";
element.style.justifyContent = "center";
element.style.alignItems = "center";
element.style.textAlign = "center";
element.style.fontWeight = "bold";
var element=document.getElementById("poison-score");
element.innerHTML = `
    <p><strong>CACC评分:</strong> ${result["CACC"]}</p>
    <p><strong>ASR评分:</strong> ${result["ASR"]}</p>
    <p><strong>ALDp分:</strong> ${result["ALDp"]}</p>
    <p><strong>投毒评分:</strong> ${((0.175*result["CACC"]+0.1376*result['ASR']+0.1429*result['ALDp'])/0.4555)}</p>
`;

element.style.fontSize = "1rem";
element.style.marginTop = "1rem";
element.style.display = "flex";
element.style.flexDirection = "column";
element.style.justifyContent = "center";
element.style.alignItems = "center";
element.style.textAlign = "center";
element.style.fontWeight = "bold";
var element_score=document.getElementById("final-score");

element_score.innerHTML=`<p><strong> 评估总分：</strong>${result['Score']}</p>`;


// 计算综合得分
var adversarialScore = (0.1541 * parseFloat(result["Tstd"]) + 0.1375 * parseFloat(result["Tsize"])) / 0.2916;
var backdoorScore = (0.0752 * parseFloat(result["NTE"]) + 0.0947 * parseFloat(result["RGB"]) + 0.083 * parseFloat(result["RIC"])) / 0.2582;
var poisonScore = (0.175 * parseFloat(result["CACC"]) + 0.1376 * parseFloat(result["ASR"]) + 0.1429 * parseFloat(result["ALDp"])) / 0.4555;
chart1=[];
// 添加到雷达图的数据中
chart1.push({ item: "对抗综合得分", score: adversarialScore });
chart1.push({ item: "后门综合得分", score: backdoorScore });
chart1.push({ item: "投毒综合得分", score: poisonScore });

// 创建雷达图
const chart_final= new G2.Chart({
    container: 'final-report-1',
    autoFit: true,
    width:500,
    height: 500,
    padding: [20, 20, 95, 20]
    });
    


chart_final.data(chart1);
chart_final.scale('score',{
    min:0,
    max:100
});
chart_final.coordinate('polar',{
    radius:0.8
});
chart_final.tooltip({
    shared:true,
    showCrosshairs:true,
    crosshairs:{
        line:{
            style:{
                lineDash:[4,4],
                stroke:'#333'
            }
        }
    }
});

chart_final.axis('item', {
line: null,
tickLine: null,
grid: {
    line: {
    style: {
        lineDash: null
    }
    }
}
});
chart_final.axis('score', {
line: null,
tickLine: null,
grid: {
    type: 'polygon',
    line: {
    style: {
        lineDash: null
    }
    }
}
});

chart_final.legend('user', {
marker: 'circle'
});


chart_final
    .area()
    .position('item*score')
    .color('orange', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    .style({ opacity: 0.5 });

    
chart_final
.line()
.position('item*score')
.color('orange')
.size(2)
.style({
    fillOpacity:0.2
});

chart_final
.point()
.position('item*score')
.color('user')
.shape('circle')
.size(4)
.style({
    stroke: '#fff',
    lineWidth: 1,
    fillOpacity: 1
});

chart_final.render();