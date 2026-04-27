/**
 * 基本特徴量エクスポート（香取市・十日町市対応版）
 *
 * 出力特徴量:
 *   - NDVI_flood : 4〜5月 NDVI 中央値（代かき・湛水期。水田は低い）
 *   - NDVI_grow  : 7〜9月 NDVI 中央値（水稲生育最盛期。水田は高い）
 *   - VH_min     : 年間 VH 最小値（湛水時に最低 → 水田識別の最重要特徴）
 *   - VH_winter  : 12〜2月 VH 中央値（冬期基準値）
 *   - point_lat  : 筆ポリゴン代表点緯度
 *   - point_lng  : 筆ポリゴン代表点経度
 *
 * 使い方:
 *   1. REGION を切り替えて2地域分それぞれ実行
 *      'katori' → katori_features_base_2023.csv
 *      'tokamachi' → tokamachi_features_base_2023.csv
 *   2. Tasks タブから CSV を Google Drive にエクスポート
 */

// ============================================================
// ★ここを切り替えて実行する
// ============================================================
var REGION = 'katori';
// 'katori' / 'tokamachi'

// ============================================================
// 設定
// ============================================================
var CONFIGS = {
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
// Sentinel-2 NDVI
// ============================================================
var s2_ndvi = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(shp.geometry())
  .filterDate(year + '-01-01', year + '-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .map(function(img) {
    return img.normalizedDifference(['B8', 'B4'])
              .rename('NDVI')
              .copyProperties(img, ['system:time_start']);
  });

// NDVI_flood: 4〜5月中央値（代かき・湛水期）
var ndvi_flood = s2_ndvi
  .filterDate(year + '-04-01', year + '-06-01')
  .median()
  .rename('NDVI_flood');

// NDVI_grow: 7〜9月中央値（水稲生育最盛期）
var ndvi_grow = s2_ndvi
  .filterDate(year + '-07-01', year + '-10-01')
  .median()
  .rename('NDVI_grow');

// ============================================================
// Sentinel-1 VH
// ============================================================
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(shp.geometry())
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
  .select('VH');

// VH_min: 年間最小値（湛水時に最低値）
var vh_min = s1
  .filterDate(year + '-01-01', year + '-12-31')
  .min()
  .rename('VH_min');

// VH_winter: 12〜2月中央値（冬期基準値）
// 前年12月〜当年2月を合算して取得
var vh_winter = s1
  .filterDate((year - 1) + '-12-01', year + '-03-01')
  .median()
  .rename('VH_winter');

// ============================================================
// 代表点座標を属性として追加する関数
// ============================================================
function addCentroid(feature) {
  var c = feature.geometry().centroid(10);
  return feature
    .set('point_lat', c.coordinates().get(1))
    .set('point_lng', c.coordinates().get(0));
}
var shp_with_pt = shp.map(addCentroid);

// ============================================================
// 全特徴量を統合して筆ポリゴンごとに集計
// ============================================================
var feat_img = ee.Image.cat([ndvi_flood, ndvi_grow, vh_min, vh_winter]);

var result = feat_img.reduceRegions({
  collection: shp_with_pt,
  reducer:    ee.Reducer.mean(),
  scale:      10,
  tileScale:  4,
});

// ============================================================
// エクスポート
// ============================================================
Export.table.toDrive({
  collection:     result,
  description:    REGION + '_features_base_' + year,
  fileNamePrefix: REGION + '_features_base_' + year,
  fileFormat:     'CSV',
  selectors: [
    cfg.id_col,
    'point_lat',
    'point_lng',
    'NDVI_flood',
    'NDVI_grow',
    'VH_min',
    'VH_winter',
  ],
});

// ============================================================
// デバッグ用出力
// ============================================================
print('対象地域:', REGION);
print('ポリゴン数:', shp.size());
print('4-5月 S2シーン数:', s2_ndvi.filterDate(year+'-04-01', year+'-06-01').size());
print('7-9月 S2シーン数:', s2_ndvi.filterDate(year+'-07-01', year+'-10-01').size());
print('年間 S1シーン数:', s1.filterDate(year+'-01-01', year+'-12-31').size());
print('冬期 S1シーン数:', s1.filterDate((year-1)+'-12-01', year+'-03-01').size());
