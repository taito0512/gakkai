/**
 * 時系列特徴量エクスポート v2（特徴量の質向上版）
 *
 * 追加する特徴量:
 *   【統計量】
 *   - NDVI_max    : 年間NDVI最大値
 *   - NDVI_min    : 年間NDVI最小値（耕起時に急落→畑の識別に最有力）
 *   - NDVI_range  : NDVI_max - NDVI_min（年間変動幅）
 *   - NDVI_std    : 年間NDVI標準偏差
 *   - NDVI_mean   : 年間NDVI平均
 *   - VH_std      : 年間VH標準偏差
 *   - VH_mean     : 年間VH平均
 *   - NDWI_mean   : 年間NDWI平均（水分指数）
 *
 *   【月別NDVI（作付けサイクルを直接捉える）】
 *   - NDVI_apr    : 4月中央値（春耕起・播種期）
 *   - NDVI_jun    : 6月中央値（初夏生育期）
 *   - NDVI_aug    : 8月中央値（夏生育ピーク）
 *   - NDVI_oct    : 10月中央値（秋収穫・耕起期）
 *
 * 使い方:
 *   1. REGION を切り替えて3地域分それぞれ実行
 *   2. Tasks タブから CSV をGoogle Driveにエクスポート
 *   3. ファイル名例: tsukubamirai_features_v2_2023.csv
 */

// ============================================================
// ★ここを切り替えて5地域分実行する
// ============================================================
var REGION = 'katori';  // 'tsukubamirai' / 'inashiki' / 'kasama' / 'katori' / 'tokamachi'

// ============================================================
// 設定
// ============================================================
var CONFIGS = {
  tsukubamirai: {
    shp:    'projects/my-project-taito/assets/2023_tukubamirai_shp',
    year:   2023,
    id_col: 'polygon_uu',
  },
  inashiki: {
    shp:    'projects/my-project-taito/assets/2022_inashiki_shp',
    year:   2023,
    id_col: 'polygon_uu',
  },
  kasama: {
    shp:    'projects/my-project-taito/assets/2022_kasama_shp',
    year:   2023,
    id_col: 'polygon_uu',
  },
  katori: {
    shp:    'projects/my-project-taito/assets/2023_katori_shp',
    year:   2023,
    id_col: 'polygon_uu',
  },
  tokamachi: {
    shp:    'projects/my-project-taito/assets/2023_tokamachi_shp',
    year:   2023,
    id_col: 'polygon_uu',
  },
};

var cfg  = CONFIGS[REGION];
var year = cfg.year;
var shp  = ee.FeatureCollection(cfg.shp);

// ============================================================
// Sentinel-2 年間統計
// ============================================================
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(shp.geometry())
  .filterDate(year + '-01-01', year + '-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .map(function(img) {
    var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
    var ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI');
    return img.addBands([ndvi, ndwi])
              .select(['NDVI', 'NDWI'])
              .copyProperties(img, ['system:time_start']);
  });

var ndvi_col   = s2.select('NDVI');
var ndvi_mean  = ndvi_col.mean().rename('NDVI_mean');
var ndvi_max   = ndvi_col.max().rename('NDVI_max');
var ndvi_min   = ndvi_col.min().rename('NDVI_min');
var ndvi_std   = ndvi_col.reduce(ee.Reducer.stdDev()).rename('NDVI_std');
var ndvi_range = ndvi_max.subtract(ndvi_min).rename('NDVI_range');
var ndwi_mean  = s2.select('NDWI').mean().rename('NDWI_mean');

// ============================================================
// 月別NDVI中央値（作付けサイクルを捉える）
// ============================================================
function monthlyNDVI(month, name) {
  var start = ee.Date.fromYMD(year, month, 1);
  var end   = start.advance(1, 'month');
  var med   = ndvi_col.filterDate(start, end).median();
  // 当月に雲なし画像がない場合は前後1ヶ月まで拡張
  var fallback = ndvi_col
    .filterDate(start.advance(-1, 'month'), end.advance(1, 'month'))
    .median();
  return ee.Image(ee.Algorithms.If(
    ndvi_col.filterDate(start, end).size().gt(0),
    med,
    fallback
  )).rename(name);
}

var ndvi_apr = monthlyNDVI(4,  'NDVI_apr');
var ndvi_jun = monthlyNDVI(6,  'NDVI_jun');
var ndvi_aug = monthlyNDVI(8,  'NDVI_aug');
var ndvi_oct = monthlyNDVI(10, 'NDVI_oct');

// ============================================================
// Sentinel-1 VH 年間統計
// ============================================================
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(shp.geometry())
  .filterDate(year + '-01-01', year + '-12-31')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .select('VH');

var vh_mean = s1.mean().rename('VH_mean');
var vh_std  = s1.reduce(ee.Reducer.stdDev()).rename('VH_std');

// ============================================================
// 全特徴量をまとめて筆ポリゴンごとに集計
// ============================================================
var feat_img = ee.Image.cat([
  ndvi_mean, ndvi_max, ndvi_min, ndvi_range, ndvi_std,
  ndwi_mean,
  ndvi_apr, ndvi_jun, ndvi_aug, ndvi_oct,
  vh_mean, vh_std,
]);

var result = feat_img.reduceRegions({
  collection: shp,
  reducer:    ee.Reducer.mean(),
  scale:      10,
  tileScale:  4,
});

// ============================================================
// エクスポート
// ============================================================
Export.table.toDrive({
  collection:     result,
  description:    REGION + '_features_v2_' + year,
  fileNamePrefix: REGION + '_features_v2_' + year,
  fileFormat:     'CSV',
  selectors: [
    cfg.id_col,
    'NDVI_mean', 'NDVI_max', 'NDVI_min', 'NDVI_range', 'NDVI_std',
    'NDWI_mean',
    'NDVI_apr', 'NDVI_jun', 'NDVI_aug', 'NDVI_oct',
    'VH_mean', 'VH_std',
  ],
});

print('対象地域:', REGION);
print('ポリゴン数:', shp.size());
print('Sentinel-2 シーン数:', s2.size());
print('Sentinel-1 シーン数:', s1.size());
print('4月シーン数:', ndvi_col.filterDate(year+'-04-01', year+'-05-01').size());
print('6月シーン数:', ndvi_col.filterDate(year+'-06-01', year+'-07-01').size());
print('8月シーン数:', ndvi_col.filterDate(year+'-08-01', year+'-09-01').size());
print('10月シーン数:', ndvi_col.filterDate(year+'-10-01', year+'-11-01').size());
