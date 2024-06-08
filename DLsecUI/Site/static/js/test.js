// 使用 fetch API 获取 CSV 文件
fetch(result_file)
    .then(response => response.text())
    .then(csvString => {
        // 使用 Papa Parse 将 CSV 字符串解析为 JSON
        const data = Papa.parse(csvString, { header: true, dynamicTyping: true }).data;

        // 创建雷达图
        const chart = new G2.Chart({
            container: 'statistic-adversarial',
            autoFit: true,
            height: 500,
        });

        chart.data(data);
        chart.scale('score', {
            min: 0,
            max: 100
        });
        chart.coordinate('polar', {
            radius: 0.8
        });
        chart.tooltip({
            shared: true,
            showCrosshairs: true,
            crosshairs: {
                line: {
                    style: {
                        lineDash: [4, 4],
                        stroke: '#333'
                    }
                }
            }
        });

        chart.axis('item', {
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
        chart.axis('score', {
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

        chart.legend('user', {
            marker: 'circle'
        });

        chart
            .line()
            .position('item*score')
            .color('user')
            .size(2);

        chart
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

        chart.render();
    });