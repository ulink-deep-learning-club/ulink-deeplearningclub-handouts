<template>
  <n-config-provider :theme="theme">
    <n-layout class="document-viewer">
      <n-layout-header class="viewer-header" bordered>
        <n-space align="center" justify="space-between">
          <n-h1 style="margin: 0;">Document Viewer</n-h1>
          <n-space>
            <n-button quaternary circle @click="toggleTheme">
              <template #icon>
                <n-icon :component="theme === darkTheme ? Sun : Moon" />
              </template>
            </n-button>
          </n-space>
        </n-space>
      </n-layout-header>
      
      <n-layout has-sider class="viewer-container">
        <n-layout-sider
          class="navigation-panel"
          width="300"
          bordered
          show-trigger="bar"
          collapse-mode="width"
          :collapsed-width="0"
        >
          <DocumentTree
            :documents="documents"
            :selected-document="selectedDocument"
            @select-document="handleDocumentSelect"
          />
        </n-layout-sider>
        
        <n-layout-content class="content-panel">
          <DocumentViewer
            :selectedDocument="selectedDocument"
            :loading="loading"
            :dark="isDarkTheme"
          />
        </n-layout-content>
      </n-layout>
    </n-layout>
  </n-config-provider>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { darkTheme, NConfigProvider, NLayout, NLayoutHeader, NLayoutSider, NLayoutContent, NSpace, NH1, NButton, NIcon } from 'naive-ui'
import { Sun, Moon } from '@vicons/tabler'
import DocumentTree from './components/DocumentTree.vue'
import DocumentViewer from './components/DocumentViewer.vue'

// Reactive state
const documents = ref([])
const selectedDocument = ref(null)
const loading = ref(false)
const isDarkTheme = ref(false)

// Load table of contents from JSON file
const loadTableOfContents = async () => {
  try {
    const response = await fetch('./dist/toc.json')
    if (!response.ok) {
      throw new Error('Failed to load table of contents')
    }
    const data = await response.json()
    documents.value = data.documents || []
    
    // Select first document by default
    if (documents.value.length > 0) {
      selectedDocument.value = documents.value[0]
    }
  } catch (error) {
    console.error('Error loading table of contents:', error)
    // Fallback to sample data for development
    documents.value = [
      {
        id: 'sample',
        title: 'Sample Document',
        path: 'sample.html',
        description: 'Sample document for testing'
      }
    ]
    selectedDocument.value = documents.value[0]
  }
}

const handleDocumentSelect = (document) => {
  loading.value = true
  selectedDocument.value = document
  
  // Simulate loading delay for better UX
  setTimeout(() => {
    loading.value = false
  }, 300)
}

const toggleTheme = () => {
  isDarkTheme.value = !isDarkTheme.value
}

const theme = computed(() => isDarkTheme.value ? darkTheme : null)

onMounted(() => {
  loadTableOfContents()
})
</script>

<style scoped>
.document-viewer {
  height: 100%;
}

.viewer-header {
  padding: 1rem 2rem;
}

.viewer-container {
  height: calc(100vh - 73px);
}

.navigation-panel {
  height: 100%;
}

.content-panel {
  height: 100%;
  overflow: hidden;
}
</style>