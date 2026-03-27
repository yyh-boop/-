// pages/index/index.js
var API_URL = 'http://127.0.0.1:5000/api/predict';
var STORAGE_KEY = 'ai_detect_history';
var MAX_SESSIONS = 50;

// 保存一次检测会话到本地存储
function saveHistory(items) {
  if (!items || !items.length) return;
  try {
    var raw = wx.getStorageSync(STORAGE_KEY);
    var sessions = raw ? JSON.parse(raw) : [];
    var now = new Date();
    var pad = function (n) { return String(n).padStart(2, '0'); };
    var timeStr = now.getFullYear() + '-' + pad(now.getMonth() + 1) + '-' + pad(now.getDate())
      + ' ' + pad(now.getHours()) + ':' + pad(now.getMinutes()) + ':' + pad(now.getSeconds());
    sessions.unshift({ id: Date.now(), time: timeStr, items: items });
    if (sessions.length > MAX_SESSIONS) sessions.splice(MAX_SESSIONS);
    wx.setStorageSync(STORAGE_KEY, JSON.stringify(sessions));
  } catch (e) {}
}

Page({
  data: {
    imageList: [],   // [{ path, name, status, label, labelClass, aiText, realText, aiPercent, realPercent, errMsg }]
    loading: false,
    btnText: '开始检测',
    doneCount: 0,
    totalCount: 0
  },

  // 选择图片（追加，最多9张）
  chooseImage: function () {
    var that = this;
    var remain = 9 - this.data.imageList.length;
    if (remain <= 0) {
      wx.showToast({ title: '最多选9张图片', icon: 'none' });
      return;
    }
    wx.chooseMedia({
      count: remain,
      mediaType: ['image'],
      sourceType: ['album', 'camera'],
      success: function (res) {
        var newItems = res.tempFiles.map(function (f) {
          var parts = f.tempFilePath.split('/');
          return {
            path: f.tempFilePath,
            name: parts[parts.length - 1] || '图片',
            status: 'pending',   // pending | loading | done | error
            label: '',
            labelClass: 'tag-pending',
            aiText: '--',
            realText: '--',
            aiPercent: 0,
            realPercent: 0,
            errMsg: ''
          };
        });
        that.setData({
          imageList: that.data.imageList.concat(newItems)
        });
      }
    });
  },

  // 删除某张图片
  removeImage: function (e) {
    var idx = e.currentTarget.dataset.idx;
    var list = this.data.imageList.slice();
    list.splice(idx, 1);
    this.setData({ imageList: list });
  },

  // 清空所有图片
  clearAll: function () {
    this.setData({ imageList: [], doneCount: 0, totalCount: 0 });
  },

  // 并行检测所有图片
  analyzeAll: function () {
    var that = this;
    var list = this.data.imageList;
    if (!list.length) {
      wx.showToast({ title: '请先选择图片', icon: 'none' });
      return;
    }

    var total = list.length;
    var doneCount = 0;

    // 全部置为 loading
    var loadingList = list.map(function (item) {
      return Object.assign({}, item, { status: 'loading', label: '', labelClass: 'tag-pending',
        aiText: '--', realText: '--', aiPercent: 0, realPercent: 0, errMsg: '' });
    });
    this.setData({ imageList: loadingList, loading: true, btnText: '检测中...', doneCount: 0, totalCount: total });

    // 收集每张图片结果，用于保存历史
    var sessionItems = new Array(list.length);

    // 包装 uploadOne，额外记录结果
    function uploadOneWithRecord(item, idx) {
      return new Promise(function (resolve) {
        wx.uploadFile({
          url: API_URL,
          filePath: item.path,
          name: 'image',
          timeout: 1000000,
          success: function (res) {
            var data = null;
            try { data = JSON.parse(res.data); } catch (e) { data = null; }

            doneCount++;
            var key = 'imageList[' + idx + ']';

            if (!data || data.error) {
              sessionItems[idx] = { name: item.name, label: '失败', error: (data && data.error) || '返回格式异常' };
              that.setData(Object.fromEntries([
                [key + '.status', 'error'],
                [key + '.errMsg', sessionItems[idx].error],
                [key + '.labelClass', 'tag-error'],
                ['doneCount', doneCount]
              ]));
            } else {
              var ai   = Number(data.ai_probability   || 0);
              var real = Number(data.real_probability || 0);
              var cls  = data.predicted_label || '未知';
              var lc   = cls === 'AI生成' ? 'tag-ai' : (cls === '真实图片' ? 'tag-real' : 'tag-pending');
              sessionItems[idx] = { name: item.name, label: cls, aiPercent: Math.round(ai * 100), realPercent: Math.round(real * 100) };
              that.setData(Object.fromEntries([
                [key + '.status', 'done'],
                [key + '.label', cls],
                [key + '.labelClass', lc],
                [key + '.aiText',   (ai   * 100).toFixed(2) + '%'],
                [key + '.realText', (real * 100).toFixed(2) + '%'],
                [key + '.aiPercent',   Math.round(ai   * 100)],
                [key + '.realPercent', Math.round(real * 100)],
                ['doneCount', doneCount]
              ]));
            }
            resolve();
          },
          fail: function (err) {
            doneCount++;
            var key = 'imageList[' + idx + ']';
            sessionItems[idx] = { name: item.name, label: '失败', error: '请求失败：' + (err.errMsg || '网络错误') };
            that.setData(Object.fromEntries([
              [key + '.status', 'error'],
              [key + '.errMsg', sessionItems[idx].error],
              [key + '.labelClass', 'tag-error'],
              ['doneCount', doneCount]
            ]));
            resolve();
          }
        });
      });
    }

    // 并行发送所有请求
    var tasks = list.map(function (item, idx) {
      return uploadOneWithRecord(item, idx);
    });

    Promise.all(tasks).then(function () {
      // 保存历史记录
      saveHistory(sessionItems.filter(Boolean));
      that.setData({ loading: false, btnText: '重新检测' });
      wx.showToast({ title: '检测完成', icon: 'success', duration: 1500 });
    });
  }
});
