<template>
  <div class="document-viewer">
    <n-card v-if="selectedDocument" class="viewer-toolbar" size="small" :bordered="false">
      <template #header>
        <n-space vertical :size="0">
          <n-h2 style="margin: 0; white-space: nowrap;">
            <n-ellipsis style="font-size: 24px; font-weight: bold;">
            {{ selectedDocument.title }}
          </n-ellipsis>
          </n-h2>

          <n-text v-if="selectedDocument.description" depth="3">
            <n-ellipsis>
            {{ selectedDocument.description }}
          </n-ellipsis>
          </n-text>
        </n-space>
      </template>
      <template #header-extra>
        <n-space>
          
          <n-tooltip trigger="hover">
            <template #trigger>
              <n-button @click="refreshDocument" size="small" quaternary>
                <template #icon>
                  <n-icon><refresh /></n-icon>
                </template>
              </n-button>
            </template>
            Refresh document
          </n-tooltip>

          <n-tooltip trigger="hover">
            <template #trigger>
              <n-button @click="openInNewTab" size="small" quaternary>
                <template #icon>
                  <n-icon><box-multiple /></n-icon>
                </template>
              </n-button>
            </template>
            Open in new tab
          </n-tooltip>

          <n-tooltip trigger="hover">
            <template #trigger>
              <n-button @click="printDocument" size="small" quaternary>
                <template #icon>
                  <n-icon><printer /></n-icon>
                </template>
              </n-button>
            </template>
            Print document
          </n-tooltip>
        </n-space>
      </template>
    </n-card>
    
    <div class="viewer-content">
      <div v-if="iframeLoading" class="loading-state">
        <n-spin size="large" />
        <n-text depth="3">Loading document...</n-text>
      </div>
      
      <div v-else-if="!selectedDocument" class="empty-state">
        <n-empty description="No Document Selected">
          <template #extra>
            <n-text depth="3">Select a document from the navigation panel to view it here.</n-text>
          </template>
        </n-empty>
      </div>
      
      <n-card class="document-card" :bordered="false" content-style="padding: 0; border-top: 1px solid #e0e0e0;">
        <iframe
          ref="documentFrame"
          :src="documentUrl"
          class="document-iframe"
          @load="onIframeLoad"
          @error="onIframeError"
        ></iframe>
      </n-card>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import { Refresh, BoxMultiple, Printer } from "@vicons/tabler"
import { NCard, NButton, NIcon, NSpin, NText, NH2, NEmpty, NSpace, NTooltip, NEllipsis } from 'naive-ui'

const props = defineProps({
  selectedDocument: {
    type: Object,
    default: null
  },
  dark: {
    type: Boolean,
    default: false
  }
})

const updateFrameStyle = (isDark) => {
  const doc = documentFrame.value;
  if (doc){
    doc.contentWindow.document.body.style.filter = isDark ? "invert(1) hue-rotate(180deg)" : "";
    doc.contentWindow.document.body.style.background = isDark ? "black" : "#fff";
  }
}
watch(() => props.dark, updateFrameStyle)

const documentFrame = ref(null)
const iframeLoading = ref(false)
const iframeError = ref(false)

const documentUrl = computed(() => {
  if (!props.selectedDocument || !props.selectedDocument.path) {
    return ''
  }
  
  // Construct the full URL for the document
  // Assuming documents are in the dist folder
  return `./dist/${props.selectedDocument.path}`
})

const refreshDocument = () => {
  if (documentFrame.value && props.selectedDocument) {
    iframeLoading.value = true
    iframeError.value = false
    documentFrame.value.src = documentUrl.value
  }
}

const openInNewTab = () => {
  if (props.selectedDocument && props.selectedDocument.path) {
    window.open(documentUrl.value, '_blank')
  }
}

const printDocument = () => {
  if (documentFrame.value) {
    documentFrame.value.contentWindow.print()
  }
}

const onIframeLoad = () => {
  // Apply custom styles to the loaded document
  applyDocumentStyles()

  iframeLoading.value = false
  iframeError.value = false
  updateFrameStyle(props.dark)
}

const applyDocumentStyles = () => {
  if (!documentFrame.value || !documentFrame.value.contentDocument) {
    return
  }
  
  try {
    const iframeDoc = documentFrame.value.contentDocument
    const style = iframeDoc.createElement('style')
    style.textContent = `
      body {
        // background-color: #ffffff !important;
        padding: 3rem 2rem !important;
        margin: 0 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif !important;
        line-height: 1.6 !important;
      }

      /* Fix the incorrect border radius generated in tcolorbox */
      .tcolorboxtitle {
        border-radius: 9px 9px 0 0 !important;
      }

      /* Fix the incorrect layout of columns */
      .multicols {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        justify-content: center;
      }
    `
    iframeDoc.head.appendChild(style)
  } catch (error) {
    console.warn('Could not apply custom styles to iframe document:', error)
  }
}

const onIframeError = () => {
  iframeLoading.value = false
  iframeError.value = true
}

// Watch for document changes
watch(() => props.selectedDocument, () => {
  if (props.selectedDocument) {
    iframeLoading.value = true
    iframeError.value = false
  }
})
</script>

<style scoped>
.document-viewer {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.viewer-toolbar {
  margin-bottom: 1rem;
}

.viewer-content {
  flex: 1;
  position: relative;
  overflow: hidden;
}

.document-card {
  height: 100%;
  background-color: var(--n-color-embedded);
}

.document-iframe {
  width: 100%;
  height: 100%;
  border: none;
  background-color: white;
  opacity: v-bind('iframeLoading ? 0 : 1');
  transition: ease-out opacity 0.15s;
}

.loading-state,
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  gap: 1rem;
}

.loading-state {
  color: var(--n-text-color-3);
}

.empty-state {
  color: var(--n-text-color-3);
}
</style>