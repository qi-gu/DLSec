

        
    var scores = result.scores;
    var total_scores = result.total_scores;
    

    // 创建雷达图
    const chart_backdoor = new G2.Chart({
        container: 'statistic-backdoor',
        autoFit: true,
        width:500,
        height: 500,
        padding: [20, 20, 95, 20]
        });
        
    console.log(scores)
    console.log(scores);
   
    var chartData = Object.keys(scores).map(key => {
        var score=parseFloat(scores[key]);
        if (isNaN(score)){
            console.log("score is NaN")
            score=1;
        }
        return {
            item:key,
            score:score*100
        }
    });
    

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
    .line()
    .position('item*score')
    .color('user')
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
   