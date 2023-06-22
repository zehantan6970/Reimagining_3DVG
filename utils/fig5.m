figure;
hold on;

% Data
x1 = [220, 694, 2048, 860];
y1 = [2.069,3.34,5.56,3.634];
x2 = [996,715,966,860];
y2 = [7.818,5.543,3.862,3.634];

% Scatter plot and text labels
sz = 200;
scatter(x1(1:end-1), y1(1:end-1), sz, 'b', 'filled');
scatter(x2, y2, sz, 'r', 'filled');
scatter(x1(end), y1(end), sz, 'g', 'filled');

% Text labels
str1 = {'OFA-tiny(Acc:0.536)', 'DINO(Acc:0.733)', ''};
str2 = {'', 'EDA(Acc:0.132)', 'Multi-View(Acc:0.310)', ' '};
text(x1(1:end-1), y1(1:end-1)+0.12, str1, 'FontSize', 30);
text(x2, y2+0.20, str2, 'FontSize', 30);

str3={'UNINEXT(Acc:0.603)'};
text(1550, 5.56+0.20, str3, 'FontSize', 30);

str4={'ScanRefer(Acc:0.038)'};
text(996, 7.818+0.05, str4, 'FontSize', 30);
% Special color for ours
text(x1(end)-10, y1(end)+0.12, 'Ours(Acc:', 'FontSize', 30);
text(x1(end)+230, y1(end)+0.12, '0.761', 'FontSize', 30, 'Color', 'b');
text(x1(end)+360, y1(end)+0.12, '/', 'FontSize', 30);
text(x1(end)+380, y1(end)+0.12, '0.590', 'FontSize', 30, 'Color', 'r');
text(x1(end)+510, y1(end)+0.12, ')', 'FontSize', 30);

% Dummy plots for legend
p1 = scatter(nan, nan, sz, 'b', 'filled');
p2 = scatter(nan, nan, sz, 'r', 'filled');
p3 = scatter(nan, nan, sz, 'g', 'filled');

% Legend
l1 = legend([p1 p2 p3], {'2DVG', '3DVG', 'Ours'}, 'FontSize', 35, 'Location', 'NorthWest');

% Labels
xlabel('Moudle file-size', 'FontSize', 50);
ylabel('Inference Time(s/sample)', 'FontSize', 50);

% Axis limit
xlim([100 2050]);

% Increase size of axis tick labels
ax = gca;
ax.XAxis.FontSize = 25;
ax.YAxis.FontSize = 25;

% Grid
grid on;
box on;

% Legend box
set(l1,'Box','on');
